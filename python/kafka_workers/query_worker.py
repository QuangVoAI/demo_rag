"""Query worker for the Nhatrovn read-only room assistant."""
import asyncio
import signal
import json
import sys
import os
import time
from pathlib import Path

# Fix Windows console encoding for Unicode emoji
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.append(str(Path(__file__).parent.parent))

from confluent_kafka import Consumer, Producer, KafkaError
from rich.console import Console

from kafka_workers.kafka_config import (
    BROKERS, GROUP_QUERY,
    TOPIC_QUERY_REQUEST, TOPIC_QUERY_RESPONSE,
    serialize, deserialize,
)
from agents.graph import run_streaming

PIPELINE_TIMEOUT_SECONDS = 120  # Max 2 phút cho một request

console = Console(force_terminal=True, safe_box=True)
running = True


def signal_handler(sig, frame):
    global running
    console.print("[yellow]🛑 Shutting down query worker...[/]")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def create_consumer() -> Consumer:
    return Consumer({
        "bootstrap.servers": BROKERS,
        "group.id": GROUP_QUERY,
        "auto.offset.reset": "latest",
        "enable.auto.commit": True,
    })


def create_producer() -> Producer:
    return Producer({
        "bootstrap.servers": BROKERS,
        "acks": "all",
    })


def run_worker():
    """Main loop for Nhatrovn read-only query processing."""
    console.print("[bold cyan]Starting Nhatrovn Query Worker (read-only room assistant)...[/]")
    console.print(f"[dim]  Listening on: {TOPIC_QUERY_REQUEST}[/]")
    console.print(f"[dim]  Publishing to: {TOPIC_QUERY_RESPONSE}[/]")
    console.print("[dim]  Pipeline: parse -> state -> read-only tools -> grounding -> final response[/]")

    # Pre-load models trước khi nhận query (tránh cold start timeout)
    from agents.model_registry import warmup
    warmup()

    consumer = create_consumer()
    producer = create_producer()
    consumer.subscribe([TOPIC_QUERY_REQUEST])

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    query_count = 0

    while running:
        msg = consumer.poll(timeout=1.0)

        if msg is None:
            continue

        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            console.print(f"[red]❌ Kafka error: {msg.error()}[/]")
            continue

        try:
            query_event = deserialize(msg.value())
            session_id = query_event.get("session_id", "unknown")
            question = query_event.get("question", "")
            history = query_event.get("history", [])

            console.print(
                f"[cyan]💬 Query: '{question[:60]}...' (session: {session_id[:8]})[/]"
            )

            # Create stream callback that publishes tokens to Kafka
            def make_stream_callback(sid: str, prod: Producer):
                async def stream_callback(token_chunk: str):
                    """Push streaming tokens vào Kafka."""
                    stream_event = {
                        "session_id": sid,
                        "answer": token_chunk,
                        "sources": [],
                        "agent_trace": {},
                        "processing_time_ms": 0,
                        "is_final": False,
                        "chunk_type": "token",
                    }
                    prod.produce(
                        TOPIC_QUERY_RESPONSE,
                        key=sid.encode("utf-8"),
                        value=serialize(stream_event),
                    )
                    prod.poll(0)  # Trigger delivery callbacks
                return stream_callback

            stream_cb = make_stream_callback(session_id, producer)

            # Run LangGraph pipeline (timeout 2 phút)
            final_state = loop.run_until_complete(
                asyncio.wait_for(
                    run_streaming(
                        question=question,
                        history=history,
                        session_id=session_id,
                        stream_callback=stream_cb,
                    ),
                    timeout=PIPELINE_TIMEOUT_SECONDS,
                )
            )

            # Publish final response
            final_answer = final_state.get("answer", "")
            final_trace = final_state.get("agent_trace", {})
            final_response = {
                "session_id": session_id,
                "answer": final_answer,
                "intent": final_state.get("intent"),
                "session_state": final_state.get("session_state", {}),
                "listings": final_state.get("listings", []),
                "cost_estimate": final_state.get("cost_estimate"),
                "comparison": final_state.get("comparison"),
                "suggested_questions": final_state.get("suggested_questions", []),
                "sources": final_state.get("sources", []),
                "agent_trace": final_trace,
                "retrieval_confidence": final_state.get("retrieval_confidence"),
                "retrieval_low_confidence": final_state.get("retrieval_low_confidence"),
                "retrieval_feedback_retry_count": final_state.get("retrieval_feedback_retry_count", 0),
                "retrieval_attempts": final_state.get("retrieval_attempts", []),
                "processing_time_ms": final_state.get("processing_time_ms", 0),
                "is_final": True,
                "chunk_type": None,
            }
            producer.produce(
                TOPIC_QUERY_RESPONSE,
                key=session_id.encode("utf-8"),
                value=serialize(final_response),
            )
            producer.flush()

            # ── Flush Langfuse immediately for real-time dashboard ──
            try:
                from utils.observability import flush_langfuse
                flush_langfuse()
            except Exception:
                pass

            query_count += 1
            console.print(
                f"[bold green]📤 Response sent for session {session_id[:8]}... "
                f"(query #{query_count})[/]"
            )

        except Exception as e:
            console.print(f"[red]❌ Query processing error: {e}[/]")
            import traceback
            traceback.print_exc()

            try:
                error_response = {
                    "session_id": query_event.get("session_id", ""),
                    "answer": f"Lỗi xử lý: {str(e)}",
                    "intent": "GENERAL_HELP",
                    "session_state": {},
                    "listings": [],
                    "cost_estimate": None,
                    "comparison": None,
                    "suggested_questions": [],
                    "sources": [],
                    "agent_trace": {},
                    "processing_time_ms": 0,
                    "is_final": True,
                    "chunk_type": None,
                }
                producer.produce(
                    TOPIC_QUERY_RESPONSE,
                    key=query_event.get("session_id", "error").encode("utf-8"),
                    value=serialize(error_response),
                )
                producer.flush()
            except Exception:
                pass

    loop.close()
    consumer.close()
    producer.flush()

    # ── Langfuse: flush pending traces ──
    try:
        from utils.observability import flush_langfuse
        flush_langfuse()
    except Exception:
        pass

    console.print(f"[yellow]👋 Query worker stopped. Total queries: {query_count}[/]")


if __name__ == "__main__":
    run_worker()

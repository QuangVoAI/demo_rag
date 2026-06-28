from django.urls import path

from . import views


app_name = "rooms"

urlpatterns = [
    path("", views.home, name="home"),
    path("tim-phong/", views.room_list, name="room_list"),
    path("tim-phong/<str:room_id>/", views.room_detail, name="room_detail"),
    path("dat-lich/", views.book_viewing, name="book_viewing"),
    path("api/chat/identity/", views.api_chat_identity, name="api_chat_identity"),
    path("api/chat/", views.api_chat, name="api_chat"),
    path("api/chat/history/", views.api_chat_history, name="api_chat_history"),
    path("api/latency-telemetry/", views.api_latency_telemetry, name="api_latency_telemetry"),
]

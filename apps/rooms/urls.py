from django.urls import path

from . import views


app_name = "rooms"

urlpatterns = [
    path("", views.home, name="home"),
    path("tim-phong/", views.room_list, name="room_list"),
    path("tim-phong/<str:room_id>/", views.room_detail, name="room_detail"),
    path("dat-lich/", views.book_viewing, name="book_viewing"),
    path("api/health/", views.api_health, name="api_health"),
    path("api/health/deep/", views.api_health_deep, name="api_health_deep"),
    path("api/chat/", views.api_chat, name="api_chat"),
    path("api/rag/stream/", views.api_rag_stream, name="api_rag_stream"),
    path("api/rag/query/", views.api_rag_query, name="api_rag_query"),
]

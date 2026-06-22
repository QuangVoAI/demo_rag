from django.urls import path

from . import views


app_name = "listings"

urlpatterns = [
    path("", views.home, name="home"),
    path("tim-phong/", views.listing_list, name="listing_list"),
    path("tim-phong/<str:listing_id>/", views.listing_detail, name="listing_detail"),
    path("dat-lich/", views.book_viewing, name="book_viewing"),
    path("api/chat/", views.api_chat, name="api_chat"),
]

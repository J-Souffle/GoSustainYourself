from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("login/", views.login, name="login"),  # Added trailing slash
    path("logout/", views.logout, name="logout"),  # Added trailing slash
    path("callback/", views.callback, name="callback"),  # Added trailing slash
    path("predict/", views.predict_carbon, name="predict_carbon"),
]
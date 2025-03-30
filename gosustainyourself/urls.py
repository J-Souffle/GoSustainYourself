from django.urls import path
from . import views

urlpatterns = [
    path("login_view/", views.index, name="index"),
    path("login/", views.login, name="login"),  # Added trailing slash
    path("logout/", views.logout, name="logout"),  # Added trailing slash
    path("callback/", views.callback, name="callback"),  # Added trailing slash
    path("", views.home, name="home"),  # Added trailing slash
    path("predict/", views.predict_carbon, name="predict_carbon"),
    path('predict_carbon/', views.predict_carbon_view, name='carbon_footprint'),
    path("predict_recycle/", views.predict_recycle, name="predict_recycle"),  # Added trailing slash
    path("predict_recycle_page/", views.predict_recycle_view, name="predict_recycle_page"),  # Added trailing slash
    path("about/", views.about, name="about"),  # Added trailing slash
    path("messages/", views.messages, name="messages"),  # Added trailing slash
    
    
]
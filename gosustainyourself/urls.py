from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("login/", views.login, name="login"),
    path("logout/", views.logout, name="logout"),
    path("callback/", views.callback, name="callback"),
    
    # Carbon footprint paths
    path("carbon-footprint/", views.predict_carbon_view, name="carbon_footprint"),
    path("api/predict_carbon/", views.predict_carbon, name="predict_carbon_api"),  # API endpoint
    
    # Recycling paths
    path("predict_recycle_page/", views.predict_recycle_view, name="predict_recycle_page"),
    path("api/predict_recycle/", views.predict_recycle, name="predict_recycle_api"),
    
    # Other pages
    path("about/", views.about, name="about"),
    path("messages/", views.messages, name="messages"),
]
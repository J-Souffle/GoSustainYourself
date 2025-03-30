import json
import os
from urllib.parse import quote_plus, urlencode
from django.conf import settings
from django.shortcuts import redirect, render
from django.urls import reverse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from authlib.integrations.django_client import OAuth
from django.contrib.auth.decorators import login_required
from tempfile import NamedTemporaryFile
from prediction_models.carbon_prediction_service import CarbonPredictionService
from recycle_computer_vision.recycle_predictor import predict_recyclable_image
from django.contrib.auth import get_user_model, login as django_login

oauth = OAuth()
oauth.register(
    "auth0",
    client_id=settings.AUTH0_CLIENT_ID,
    client_secret=settings.AUTH0_CLIENT_SECRET,
    client_kwargs={
        "scope": "openid profile email",
    },
    server_metadata_url=f"https://{settings.AUTH0_DOMAIN}/.well-known/openid-configuration",
)

# Initialize the prediction service (you might want to do this differently in production)
carbon_predictor = CarbonPredictionService()

def index(request):
    # Render home page; if a user session exists, show welcome message, otherwise provide a login link.
    return render(
        request,
        "index.html",
        context={
            "session": request.session.get("user"),
            "pretty": json.dumps(request.session.get("user"), indent=4),
        },
    )


def login(request):
    # Build redirect_uri dynamically based on your site's URL.
    redirect_uri = request.build_absolute_uri(reverse("callback"))
    return oauth.auth0.authorize_redirect(request, redirect_uri)


@csrf_exempt
def callback(request):
    try:
        token = oauth.auth0.authorize_access_token(request)
        userinfo = token.get("userinfo")
        
        # Get or create a Django user
        email = userinfo["email"]
        User = get_user_model()
        user, _ = User.objects.get_or_create(username=email, defaults={"email": email})
        
        # Log in with Django’s session
        django_login(request, user, backend="django.contrib.auth.backends.ModelBackend")
        
        # Optional: store entire token in session if needed
        request.session["auth0_token"] = token
        return redirect("home")
    except Exception as e:
        print(f"Auth0 callback error: {str(e)}")
        return redirect("login")


def logout(request):
    # Clear local session
    request.session.clear()
    # Redirect to Auth0 logout
    return redirect(
        f"https://{settings.AUTH0_DOMAIN}/v2/logout?"
        + urlencode(
            {
                "returnTo": request.build_absolute_uri(reverse("home")), 
                "client_id": settings.AUTH0_CLIENT_ID,
            },
            quote_via=quote_plus,
        )
    )

@login_required
def predict_carbon(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        result = carbon_predictor.predict_carbon_emission(data)
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

@login_required
def predict_recyclable(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST method is allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        result = carbon_predictor.predict_recyclable(data)
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)

@login_required
@csrf_exempt
def predict_recycle(request):
    if request.method != 'POST':
        return JsonResponse({"error": "Only POST method is allowed"}, status=405)
    
    if 'file' not in request.FILES:
        return JsonResponse({"error": "No file uploaded"}, status=400)
    
    # Save the uploaded file to a temporary location.
    uploaded_file = request.FILES['file']
    with NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    
    try:
        result = predict_recyclable_image(tmp_path)
        # Optionally, remove the temporary file after prediction.
        os.remove(tmp_path)
        return JsonResponse(result)
    except Exception as e:
        os.remove(tmp_path)
        return JsonResponse({"error": str(e)}, status=400)

@login_required
def recycle_upload(request):
    """
    Renders the HTML page that lets a user upload an image
    to predict whether it is recyclable.
    """
    return render(request, "predict_recycle.html")

def home(request):
    """
    Renders the home page with a welcome message.
    """
    return render(request, "home.html", context={"session": request.session.get("user")})



def about(request):
    """
    Renders the about page.
    """
    return render(request, "about.html", context={"session": request.session.get("user")})

def messages(request):
    """
    Renders the messages page.
    """
    return render(request, "messages.html", context={"session": request.session.get("user")})

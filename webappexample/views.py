import json
from authlib.integrations.django_client import OAuth
from django.conf import settings
from django.shortcuts import redirect, render, redirect
from django.urls import reverse
from urllib.parse import quote_plus, urlencode
from django.contrib.auth.decorators import login_required

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


def index(request):

    return render(
        request,
        "index.html",
        context={
            "session": request.session.get("user"),
            "pretty": json.dumps(request.session.get("user"), indent=4),
        },
    )


def login(request):
    return oauth.auth0.authorize_redirect(
        request,
        redirect_uri=settings.AUTH0_CALLBACK_URL,
        # Add these lines to ensure proper state handling
        nonce=None,
        audience=None,
        scope='openid profile email'
    )

def callback(request):
    try:
        token = oauth.auth0.authorize_access_token(request)
        request.session['user'] = token
        return redirect('/')
    except Exception as e:
        print(f"Error in callback: {str(e)}")
        return redirect('/login')


def logout(request):
    request.session.clear()

    return redirect(
        f"https://{settings.AUTH0_DOMAIN}/v2/logout?"
        + urlencode(
            {
                "returnTo": request.build_absolute_uri(reverse("index")),
                "client_id": settings.AUTH0_CLIENT_ID,
            },
            quote_via=quote_plus,
        ),
    )

import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import environ
# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / '.env')
TEMPLATE_DIR = os.path.join(BASE_DIR, "gosustainyourself", "templates")
import environ

# Initialise environment variables
env = environ.Env()
environ.Env.read_env()  #
# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.environ.get('DEBUG', 'False') == 'True'
DEBUG = True

ALLOWED_HOSTS = [
    'localhost',
    '127.0.0.1',
    'gosustainyourself.onrender.com',  # Render 's default domain
    'gosustainyourself.tech',              # Your custom domain
    'www.gosustainyourself.tech'
    'https://gosustainyourself-496911025175.us-east4.run.app'
]

# Application definition

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    'gosustainyourself.apps.GosustainyourselfConfig',
    'django_bootstrap5',
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [TEMPLATE_DIR],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application" 


# Database
# https://docs.djangoproject.com/en/4.0/ref/settings/#databases

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# Password validation
# https://docs.djangoproject.com/en/4.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


# Internationalization
# https://docs.djangoproject.com/en/4.0/topics/i18n/

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


# Load environment definition file

ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)


# Load Auth0 application settings into memory

AUTH0_DOMAIN = os.environ.get("AUTH0_DOMAIN")
AUTH0_CLIENT_ID = os.environ.get("AUTH0_CLIENT_ID")
AUTH0_CLIENT_SECRET = os.environ.get("AUTH0_CLIENT_SECRET")

AUTH0_CALLBACK_URL = os.environ.get("AUTH0_CALLBACK_URL", "http://localhost:8000/callback/")
AUTH0_LOGOUT_URL = os.environ.get("AUTH0_LOGOUT_URL", "http://localhost:8000/")


X_FRAME_OPTIONS = 'SAMEORIGIN'  # For development only
SESSION_COOKIE_SAMESITE = 'Lax'  # Helps with Auth0 redirects
CSRF_COOKIE_SAMESITE = 'Lax'
SESSION_COOKIE_SECURE = True   # Change to True in production
CSRF_COOKIE_SECURE = True     # Change to True in production

# For local development with HTTP
SESSION_COOKIE_SECURE = False
CSRF_COOKIE_SECURE = False
SESSION_COOKIE_SAMESITE = None
CSRF_COOKIE_SAMESITE = None

AUTH0_SUCCESS_URL = "/"
LOGIN_URL = '/login'
LOGOUT_REDIRECT_URL = '/'
LOGIN_REDIRECT_URL = '/'

SESSION_ENGINE = "django.contrib.sessions.backends.db"  # Explicit session engine
SESSION_COOKIE_NAME = "gosustain_session"  # Unique cookie name
SESSION_EXPIRE_AT_BROWSER_CLOSE = True 

STATIC_URL = 'static/'
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static'), os.path.join(BASE_DIR, 'gosustainyourself', 'static')]

STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')


if os.environ.get('RENDER') != 'true':  # Render sets this automatically
    load_dotenv()

SECRET_KEY = os.environ.get('SECRET_KEY')  # Works both locally and on Render
DEBUG = os.environ.get('DEBUG', 'False') == 'True'  # Default to False



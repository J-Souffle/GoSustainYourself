import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import mongoengine
import environ
# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = os.path.join(BASE_DIR, "gosustainyourself", "templates")

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/4.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.environ.get('DJANGO_SECRET_KEY', 'your-default-secret-key-for-development')

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.environ.get('DEBUG', 'False') == 'True'
DEBUG = True

ALLOWED_HOSTS = [
    'localhost',
    '127.0.0.1',
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
    'django_bootstrap5'
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
    'default': {
        'ENGINE': 'django.db.backends.mysql',


        'NAME': 'gosustainyourself',       # Database name from Step 2
        'USER': 'root',             # MySQL username (or 'root' if no separate user)
        'PASSWORD': 'your_password',       # MySQL password
        'HOST': 'localhost',               # Usually 'localhost' or '127.0.0.1'
        'PORT': '3306',                    # Default MySQL port
        'OPTIONS': {
            'init_command': "SET sql_mode='STRICT_TRANS_TABLES'",  # Avoid MySQL warnings
        },
    }
}

# import mongoengine

# MONGO_DB_NAME = "your_database_name"
# MONGO_ATLAS_CONN_STRING = "mongodb+srv://<username>:<password>@cluster0.mongodb.net/<dbname>?retryWrites=true&w=majority"

# mongoengine.connect(
#     db=MONGO_DB_NAME,
#     host=MONGO_ATLAS_CONN_STRING
# )


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
AUTH0_CALLBACK_URL = os.environ.get("AUTH0_CALLBACK_URL", "http://127.0.0.1:8000/callback/")
print("AUTH0_DOMAIN:", AUTH0_DOMAIN)
print("AUTH0_CLIENT_ID:", AUTH0_CLIENT_ID)
print("AUTH0_CLIENT_SECRET:", AUTH0_CLIENT_SECRET)
print("AUTH0_CALLBACK_URL:", AUTH0_CALLBACK_URL)

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
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'static')]

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')



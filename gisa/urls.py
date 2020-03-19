from django.contrib import admin
from django.urls import path
from .  import views
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import TemplateView

app_name = 'gisa'

urlpatterns = [
    path('form/',views.formpage,name = 'form'),
    path('segment/',views.segment_it,name = 'segment'),
    path('live_temp/',views.livepage,name = 'live_temp'),
    # path('live/', TemplateView.as_view(template_name = 'live.html'),name = 'live')
    path('live/' , views.button_live, name = 'live'),
    path('live_segment/', views.segment_live, name = 'live_segment')

] + static(settings.MEDIA_URL, document_root = settings.MEDIA_URL)
from django.contrib import admin
from django.urls import path
from .  import views
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import TemplateView

app_name = 'gisa'

urlpatterns = [
    path('form/',views.formpage,name = 'form'),
    path('predict/', views.predict_menu,name = 'menu'),
    path('segment/',views.segment_it,name = 'segment'),
    path('live_temp/',views.livepage,name = 'live_temp'),
    # path('live/', TemplateView.as_view(template_name = 'live.html'),name = 'live')
    path('live/' , views.button_live, name = 'live'),
    path('live_segment/', views.button_segment_live, name = 'segment_live'),
    path('live_segment_temp', views.segment_live, name = 'live_segment_temp'),
    # path('segmenting_live/', views.segmenting_live, name = 'segmenting_live'),
    # path('seg_live_test/', views.seg_live_test, name = 'seg_live_test')
    

] + static(settings.MEDIA_URL, document_root = settings.MEDIA_URL)
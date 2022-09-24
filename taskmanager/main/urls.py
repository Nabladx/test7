from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('loops', views.loops, name='loops'),
    path('loops', views.generatePDF, name='generatePDF'),
]

urlpatterns += staticfiles_urlpatterns()

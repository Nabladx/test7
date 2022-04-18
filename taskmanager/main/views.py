from django.http import JsonResponse
from django.shortcuts import render, HttpResponse
from .forms import MyfileUploadForm
from .models import file_upload


def index(request):
    if 'term' in request.GET:
        qs = file_upload.objects.filter(file_name__icontains=request.GET.get('term'))
        titles = list()
        for author in qs:
            titles.append(author.file_name)
        return JsonResponse(titles, safe=False)
    if request.method == 'POST':
        form = MyfileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            test_data = form.cleaned_data['files_data']
            return HttpResponse(test_data)
        else:
            return HttpResponse('error')
    else:
        context = {
            'form': MyfileUploadForm()
        }
        return render(request, 'main/index.html', context)

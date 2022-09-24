import string
from codecs import decode
from math import fabs

from django.contrib.admin.views.decorators import staff_member_required
from django.http import JsonResponse
from django.shortcuts import render, HttpResponse
from natasha import (
    Segmenter,
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    Doc
)
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score
from .forms import MyfileUploadForm
from .models import file_upload

from django.http import FileResponse
import io
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter


@staff_member_required
def generatePDF(request, id):
    buffer = io.BytesIO()
    x = canvas.Canvas(buffer)
    x.drawString(100, 100, "Let's generate this pdf file.")
    x.showPage()
    x.save()
    buffer.seek(0)
    return FileResponse(buffer, as_attachment=True, filename='attempt1.pdf')


def loops(request):
    context = {
        'form': MyfileUploadForm()
    }
    return render(request, 'main/loops.html', context)


def index(request):
    if 'term' in request.GET:
        responce = file_upload.objects.filter(
            file_name__icontains=request.GET.get('term'))
        titles = list()
        for author in responce:
            titles.append(author.file_name)
        return JsonResponse(titles, safe=False)
    if request.method == 'POST':
        form = MyfileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            segmenter = Segmenter()
            emb = NewsEmbedding()
            morph_tagger = NewsMorphTagger(emb)
            syntax_parser = NewsSyntaxParser(emb)
            ner_tagger = NewsNERTagger(emb)
            name = request.POST.get('author')
            train_data = file_upload.objects.get(file_name=name).my_file
            train_data = Doc(str(decode(train_data.read())))
            test_data = form.cleaned_data['files_data']
            test_data = str(decode(test_data.read()))
            count_test_data = sum(
                [i.strip(string.punctuation).isalpha() for i in
                 test_data.split()])
            test_data = Doc(test_data)
            train_data.segment(segmenter)
            train_data.tag_morph(morph_tagger)
            train_data.parse_syntax(syntax_parser)
            train_data.tag_ner(ner_tagger)
            test_data.segment(segmenter)
            test_data.tag_morph(morph_tagger)
            test_data.parse_syntax(syntax_parser)
            test_data.tag_ner(ner_tagger)
            NOUN_train = 0
            ADV_train = 0
            ADP_train = 0
            ADJ_train = 0
            # PRON_train = 0
            PRON_train_obj = 0
            PRON_train_obl = 0
            PRON_train_nsubj = 0
            PRON_train_advmod = 0
            PRON_train_nmod = 0
            PRON_train_iobj = 0
            PRON_train_det = 0
            PRON_train_ccomp = 0
            NUM_train = 0
            CCONJ_train = 0
            AUX_train = 0
            PUNCT_train = 0
            SCONJ_train = 0
            # PROPN_train = 0
            PROPN_train_nsubj = 0
            PROPN_train_appos = 0
            PROPN_train_root = 0
            PROPN_train_conj = 0
            PROPN_train_iobj = 0
            PROPN_train_amod = 0
            PROPN_train_cc = 0
            PROPN_train_nmod = 0
            PROPN_train_parataxis = 0
            VERB_train = 0
            PART_train = 0
            DET_train = 0
            NOUN_test = 0
            ADV_test = 0
            ADP_test = 0
            ADJ_test = 0
            NUM_test = 0
            CCONJ_test = 0
            AUX_test = 0
            PUNCT_test = 0
            SCONJ_test = 0
            # PROPN_test = 0
            PROPN_test_nsubj = 0
            PROPN_test_appos = 0
            PROPN_test_root = 0
            PROPN_test_conj = 0
            PROPN_test_iobj = 0
            PROPN_test_amod = 0
            PROPN_test_cc = 0
            PROPN_test_nmod = 0
            PROPN_test_parataxis = 0
            VERB_test = 0
            PART_test = 0
            DET_test = 0
            # PRON_test = 0
            PRON_test_obj = 0
            PRON_test_obl = 0
            PRON_test_nsubj = 0
            PRON_test_advmod = 0
            PRON_test_nmod = 0
            PRON_test_iobj = 0
            PRON_test_det = 0
            PRON_test_ccomp = 0
            train_words = 8000
            for i in range(1, train_words):
                pos = train_data.tokens[i].pos
                rel = train_data.tokens[i].rel
                if pos == 'NOUN':
                    NOUN_train += 1
                if pos == 'ADV':
                    ADV_train += 1
                if pos == 'ADP':
                    ADP_train += 1
                if pos == 'ADJ':
                    ADJ_train += 1
                if pos == 'PRON':
                    if rel == 'obj':
                        PRON_train_obj += 1
                    if rel == 'obl':
                        PRON_train_obl += 1
                    if rel == 'nsubj':
                        PRON_train_nsubj += 1
                    if rel == 'advmod':
                        PRON_train_advmod += 1
                    if rel == 'nmod':
                        PRON_train_nmod += 1
                    if rel == 'iobj':
                        PRON_train_iobj += 1
                    if rel == 'det':
                        PRON_train_det += 1
                    if rel == 'ccomp':
                        PRON_train_ccomp += 1
                if pos == 'NUM':
                    NUM_train += 1
                if pos == 'CCONJ':
                    CCONJ_train += 1
                if pos == 'AUX':
                    AUX_train += 1
                if pos == 'PUNCT':
                    PUNCT_train += 1
                if pos == 'SCONJ':
                    SCONJ_train += 1
                if pos == 'PROPN':
                    if rel == 'nsubj':
                        PROPN_train_nsubj += 1
                    if rel == 'appos':
                        PROPN_train_appos += 1
                    if rel == 'root':
                        PROPN_train_root += 1
                    if rel == 'conj':
                        PROPN_train_conj += 1
                    if rel == 'iobj':
                        PROPN_train_iobj += 1
                    if rel == 'amod':
                        PROPN_train_amod += 1
                    if rel == 'cc':
                        PROPN_train_cc += 1
                    if rel == 'nmod':
                        PROPN_train_nmod += 1
                    if rel == 'parataxis':
                        PROPN_train_parataxis += 1
                if pos == 'VERB':
                    VERB_train += 1
                if pos == 'PART':
                    PART_train += 1
                if pos == 'DET':
                    DET_train += 1
            test_words = count_test_data
            test_words = 1700
            for i in range(1, test_words):
                pos = test_data.tokens[i].pos
                rel = test_data.tokens[i].rel
                if pos == 'NOUN':
                    NOUN_test += 1
                if pos == 'ADV':
                    ADV_test += 1
                if pos == 'ADP':
                    ADP_test += 1
                if pos == 'ADJ':
                    ADJ_test += 1
                if pos == 'PRON':
                    if rel == 'obj':
                        PRON_test_obj += 1
                    if rel == 'obl':
                        PRON_test_obl += 1
                    if rel == 'nsubj':
                        PRON_test_nsubj += 1
                    if rel == 'advmod':
                        PRON_test_advmod += 1
                    if rel == 'nmod':
                        PRON_test_nmod += 1
                    if rel == 'iobj':
                        PRON_test_iobj += 1
                    if rel == 'det':
                        PRON_test_det += 1
                    if rel == 'ccomp':
                        PRON_test_ccomp += 1
                if pos == 'NUM':
                    NUM_test += 1
                if pos == 'CCONJ':
                    CCONJ_test += 1
                if pos == 'AUX':
                    AUX_test += 1
                if pos == 'PUNCT':
                    PUNCT_test += 1
                if pos == 'SCONJ':
                    SCONJ_test += 1
                if pos == 'PROPN':
                    if rel == 'nsubj':
                        PROPN_test_nsubj += 1
                    if rel == 'appos':
                        PROPN_test_appos += 1
                    if rel == 'root':
                        PROPN_test_root += 1
                    if rel == 'conj':
                        PROPN_test_conj += 1
                    if rel == 'iobj':
                        PROPN_test_iobj += 1
                    if rel == 'amod':
                        PROPN_test_amod += 1
                    if rel == 'cc':
                        PROPN_test_cc += 1
                    if rel == 'nmod':
                        PROPN_test_nmod += 1
                    if rel == 'parataxis':
                        PROPN_test_parataxis += 1
                if pos == 'VERB':
                    VERB_test += 1
                if pos == 'PART':
                    PART_test += 1
                if pos == 'DET':
                    DET_test += 1
            diff_NOUN = fabs((NOUN_train / train_words) - \
                             (NOUN_test / test_words))
            diff_ADV = fabs((ADV_train / train_words) - \
                            (ADV_test / test_words))
            diff_ADP = fabs((ADP_train / train_words) - \
                            (ADP_test / test_words))
            diff_ADJ = fabs((ADJ_train / train_words) - \
                            (ADJ_test / test_words))
            diff_PRON_obj = fabs((PRON_train_obj / train_words) - \
                                 (PRON_test_obj / test_words))
            diff_PRON_obl = fabs((PRON_train_obl / train_words) - \
                                 (PRON_test_obl / test_words))
            diff_PRON_nsubj = fabs((PRON_train_nsubj / train_words) - \
                                   (PRON_test_nsubj / test_words))
            diff_PRON_advmod = fabs((PRON_train_advmod / train_words) - \
                                    (PRON_test_advmod / test_words))
            diff_PRON_nmod = fabs((PRON_train_nmod / train_words) - \
                                  (PRON_test_nmod / test_words))
            diff_PRON_iobj = fabs((PRON_train_iobj / train_words) - \
                                  (PRON_test_iobj / test_words))
            diff_PRON_det = fabs((PRON_train_det / train_words) - \
                                 (PRON_test_det / test_words))
            diff_PRON_ccomp = fabs((PRON_train_ccomp / train_words) - \
                                   (PRON_test_ccomp / test_words))
            diff_NUM = fabs((NUM_train / train_words) - \
                            (NUM_test / test_words))
            diff_CCONJ = fabs((CCONJ_train / train_words) - \
                              (CCONJ_test / test_words))
            diff_AUX = fabs((AUX_train / train_words) - \
                            (AUX_test / test_words))
            diff_PUNCT = fabs((PUNCT_train / train_words) - \
                              (PUNCT_test / test_words))
            diff_SCONJ = fabs((SCONJ_train / train_words) - \
                              (SCONJ_test / test_words))
            diff_PROPN_nsubj = fabs((PROPN_train_nsubj / train_words) - \
                                    (PROPN_test_nsubj / test_words))
            diff_PROPN_appos = fabs((PROPN_train_appos / train_words) - \
                                    (PROPN_test_appos / test_words))
            diff_PROPN_root = fabs((PROPN_train_root / train_words) - \
                                   (PROPN_test_root / test_words))
            diff_PROPN_conj = fabs((PROPN_train_conj / train_words) - \
                                   (PROPN_test_conj / test_words))
            diff_PROPN_iobj = fabs((PROPN_train_iobj / train_words) - \
                                   (PROPN_test_iobj / test_words))
            diff_PROPN_amod = fabs((PROPN_train_amod / train_words) - \
                                   (PROPN_test_amod / test_words))
            diff_PROPN_cc = fabs((PROPN_train_cc / train_words) - \
                                 (PROPN_test_cc / test_words))
            diff_PROPN_nmod = fabs((PROPN_train_nmod / train_words) - \
                                   (PROPN_test_nmod / test_words))
            diff_PROPN_parataxis = fabs((PROPN_train_parataxis / train_words) - \
                                        (PROPN_test_parataxis / test_words))
            diff_VERB = fabs((VERB_train / train_words) - \
                             (VERB_test / test_words))
            diff_PART = fabs((PART_train / train_words) - \
                             (PART_test / test_words))
            diff_DET = fabs((DET_train / train_words) - \
                            (DET_test / test_words))
            difference = diff_NOUN + diff_ADV + diff_ADP + diff_ADJ + \
                         diff_PRON_obj + diff_PRON_obl + diff_PRON_nsubj + \
                         diff_PRON_advmod + diff_PRON_nmod + diff_PRON_iobj + \
                         diff_PRON_det + diff_PRON_ccomp + diff_NUM + \
                         diff_CCONJ + diff_AUX + diff_PUNCT + diff_SCONJ + \
                         diff_PROPN_nsubj + diff_PROPN_appos + \
                         diff_PROPN_root + diff_PROPN_conj + diff_PROPN_iobj + \
                         diff_PROPN_amod + diff_PROPN_cc + diff_PROPN_nmod + \
                         diff_PROPN_parataxis + diff_VERB + diff_PART + diff_DET
            train = [[NOUN_train], [ADP_train], [ADJ_train],
                     [PRON_train_obj], [PRON_train_obl], [PRON_train_nsubj],
                     [PRON_train_advmod], [PRON_train_nmod], [PRON_train_iobj],
                     [PRON_train_det], [PRON_train_ccomp], [NUM_train],
                     [CCONJ_train], [AUX_train], [PUNCT_train], [SCONJ_train],
                     [PROPN_train_nsubj], [PROPN_train_appos],
                     [PROPN_train_root], [PROPN_train_conj],
                     [PROPN_train_iobj],
                     [PROPN_train_amod], [PROPN_train_cc], [PROPN_train_nmod],
                     [PROPN_train_parataxis], [VERB_train], [PART_train],
                     [DET_train]]
            test = [[NOUN_test], [ADP_test], [ADJ_test],
                    [PRON_test_obj], [PRON_test_obl], [PRON_test_nsubj],
                    [PRON_test_advmod], [PRON_test_nmod], [PRON_test_iobj],
                    [PRON_test_det], [PRON_test_ccomp], [NUM_test],
                    [CCONJ_test], [AUX_test], [PUNCT_test], [SCONJ_test],
                    [PROPN_test_nsubj], [PROPN_test_appos],
                    [PROPN_test_root], [PROPN_test_conj], [PROPN_test_iobj],
                    [PROPN_test_amod], [PROPN_test_cc], [PROPN_test_nmod],
                    [PROPN_test_parataxis], [VERB_test], [PART_test],
                    [DET_test]]
            pred_train = OneClassSVM(gamma='auto').fit(train).predict(train)
            pred_test = OneClassSVM(gamma='auto').fit(test).predict(test)
            results = accuracy_score(pred_train, pred_test)
            results = 1 - difference
            if results < 0.87:
                results *= 10000
                results = (int(results)) / 100.0
                html = "<html><body><h2>Ваш текст <ins>не</ins> принадлежит автору</body></html>"
                return HttpResponse(
                    '{} "{}" <br>Точность принадлежности равна <ins>{}</ins>%'.format(
                        html, name, results))
            else:
                results *= 10000
                results = (int(results)) / 100.0
                html = "<html><body><h2>Ваш текст принадлежит автору</body></html>"
                return HttpResponse(
                    '{} "{}" <br>Точность принадлежности равна <ins>{}</ins>%'.format(
                        html, name, results))
        else:
            return HttpResponse('error')
    else:
        context = {
            'form': MyfileUploadForm()
        }
        return render(request, 'main/index.html', context)

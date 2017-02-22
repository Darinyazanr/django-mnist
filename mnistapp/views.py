from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from utils import prediction

predictor = prediction.MnistPredictor()


def index(request):
    latest_question_list = ["test"]
    context = {'latest_question_list': latest_question_list}
    return render(request, 'mnist/index.html', context)


@csrf_exempt
def guess(request):
    uri = request.GET['image']
    num_guess = predictor.get_prediction(uri)
    return JsonResponse({'guess': num_guess})

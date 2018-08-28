from django.shortcuts import render, get_object_or_404,redirect
from django.http import HttpResponse,HttpResponseRedirect
from django.core.paginator import Paginator,EmptyPage,PageNotAnInteger
from .models import Post
from django.contrib import messages
from .forms import PostForm
# request.user.is_authenticated()
def post_home(request):
    query_set_list = Post.objects.all()#.order_by("-timestamp")
    paginator = Paginator(query_set_list,20)
    page = request.GET.get('page')
    try:
        query_set = paginator.page(page)
    except PageNotAnInteger:
        query_set = paginator.page(1)
    except EmptyPage:
        query_set = paginator.page(paginator.num_pages)
    context = {
        'title': 'HOME PAGE',
        'Post_query': query_set,
    }
    return render(request,'list_all.html',context)


def post_update(request,id=None):
    instance = get_object_or_404(Post, id=id)
    form = PostForm(request.POST or None,request.FILES or None,instance = instance)
    if form.is_valid():
        instance = form.save(commit=False)
        instance.save()
        messages.success(request, "successfully update")
        return HttpResponseRedirect(instance.get_absolute_url())
    context = {
        "title": "Detail",
        "instance": instance,
        "form":form
    }
    return render(request, "forms.html", context)


def post_create(request):
    form = PostForm(request.POST or None,request.FILES or None)
    if form.is_valid():
        instance = form.save(commit=False)
        instance.save()
        messages.success(request,"successfully create")
        return HttpResponseRedirect(instance.get_absolute_url())
    else:
        messages.error(request,"Not successfully create yet")
    context = {
        "form": form
    }
    return render(request, "forms.html", context)


def post_detail(request, id=None):
    instance = get_object_or_404(Post, id=id)
    context = {
        "title": "Detail",
        "instance": instance
    }
    return render(request, "detail.html", context)


def post_delete(request, id=None):
    instance = get_object_or_404(Post, id=id)
    instance.delete()
    messages.success(request, "successfully delete")
    return redirect('posts:list')
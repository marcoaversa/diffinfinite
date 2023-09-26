function showCarousel(carouselId) {
    // Hide all images
    var carousels = document.getElementsByClassName('selected-image');
    for(var i = 0; i < carousels.length; i++) {
        carousels[i].style.display = 'none';
    }

    document.getElementById(carouselId).style.display = 'block';
  }

function showAbstract(carouselId) {
    // Hide all images
    var carousels = document.getElementsByClassName('selected-abstract');
    for(var i = 0; i < carousels.length; i++) {
        carousels[i].style.display = 'none';
    }
    document.getElementById(carouselId).style.display = 'block';
  }

document.addEventListener("DOMContentLoaded", function() {
    setTimeout(function(){
        document.getElementById('Artery').style.display = 'none';
        document.getElementById('Artifacts').style.display = 'none';
        document.getElementById('Carcinoma').style.display = 'none';
        document.getElementById('Cartilage').style.display = 'none';
        document.getElementById('Connective').style.display = 'none';
        document.getElementById('Necrosis').style.display = 'none';
        document.getElementById('Stroma').style.display = 'none';
        document.getElementById('abstract-image').style.display = 'none';
    }, 1); // 1000 milliseconds = 1 second
  });


  document.addEventListener('DOMContentLoaded', function () {
    const buttons = document.querySelectorAll('button');

    buttons.forEach(btn => {
        btn.addEventListener('click', function () {
            buttons.forEach(button => {
                button.classList.add('is-dark'); // remove inverted class from all buttons
            });
            btn.classList.remove('is-dark'); // add inverted class to clicked button
        });
    });
});
  
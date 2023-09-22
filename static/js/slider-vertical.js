let currentSlide = 0;

function nextSlide() {
    const carousel = document.querySelector('.slider-container.is-vertical');
    const items = carousel.getElementsByClassName('slider-item');
    if (currentSlide < items.length - 1) {
        currentSlide++;
        carousel.style.transform = `translateY(-${100 * currentSlide}%)`;
    }
}

function prevSlide() {
    const carousel = document.querySelector('.slider-container.is-vertical');
    if (currentSlide > 0) {
        currentSlide--;
        carousel.style.transform = `translateY(-${100 * currentSlide}%)`;
    }
}
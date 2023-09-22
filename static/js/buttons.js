function showCarousel(carouselId) {
    const carousels = document.getElementsByClassName('hero-body');
    for (let carousel of carousels) {
        carousel.style.display = carousel.id === carouselId ? 'block' : 'none';
    }
}

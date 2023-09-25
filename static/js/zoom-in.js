document.addEventListener('DOMContentLoaded', function() {
    magnify("zoom-in-image-real", 3);
    magnify("zoom-in-image-synth", 3);
    magnify("zoom-in-image-squares", 3);
  });
  
  function magnify(imgID, zoom) {
    var img, lens, result, cx, cy;
    img = document.getElementById(imgID);
    lens = document.createElement("DIV");
    lens.setAttribute("class", "magnifier-lens");
    img.parentElement.insertBefore(lens, img);
  
    lens.style.backgroundImage = "url('" + img.src + "')";
    lens.style.backgroundRepeat = "no-repeat";
    cx = zoom;
    cy = zoom;
    lens.style.backgroundSize = (img.width * cx) + "px " + (img.height * cy) + "px";
  
    lens.addEventListener("mousemove", moveLens);
    img.addEventListener("mousemove", moveLens);
  
    function moveLens(e) {
      var pos, x, y;
      e.preventDefault();
      pos = getCursorPos(e);
      x = pos.x - (lens.offsetWidth / 2);
      y = pos.y - (lens.offsetHeight / 2);
  
      if (x > img.width - lens.offsetWidth) { x = img.width - lens.offsetWidth; }
      if (x < 0) { x = 0; }
      if (y > img.height - lens.offsetHeight) { y = img.height - lens.offsetHeight; }
      if (y < 0) { y = 0; }
  
      lens.style.left = x + "px";
      lens.style.top = y + "px";
      lens.style.backgroundPosition = "-" + (x * cx) + "px -" + (y * cy) + "px";
    }
  
    function getCursorPos(e) {
      var a, x = 0, y = 0;
      e = e || window.event;
      a = img.getBoundingClientRect();
      x = e.pageX - a.left;
      y = e.pageY - a.top;
      x = x - window.pageXOffset;
      y = y - window.pageYOffset;
      return { x: x, y: y };
    }
  }
  
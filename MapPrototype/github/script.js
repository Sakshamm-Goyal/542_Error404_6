document.addEventListener("DOMContentLoaded", function () {
    var slideIndex = 0;
    showSlides();
  
    function showSlides() {
      var i;
      var slides = document.getElementById("auto-slider").getElementsByTagName("img");
      for (i = 0; i < slides.length; i++) {
        slides[i].style.display = "none";
      }
      slideIndex++;
      if (slideIndex > slides.length) { slideIndex = 1 }
      slides[slideIndex - 1].style.display = "block";
      setTimeout(showSlides, 3000); // Change slide every 3 seconds
    }
  });
  
// Smooth scrolling for anchor links
document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();

      const targetId = this.getAttribute('href').substring(1);
      const targetElement = document.getElementById(targetId);

      if (targetElement) {
        targetElement.style.display = 'block';
        targetElement.scrollIntoView({
          behavior: 'smooth'
        });
      }
    });
  });
});

  
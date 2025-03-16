// Natural PDF custom script

// Add homepage class for styling
document.addEventListener('DOMContentLoaded', function() {
  const path = window.location.pathname;
  if (path === '/' || 
      path === '/index.html' || 
      path.endsWith('/') && !path.endsWith('/index.html') && path.split('/').filter(Boolean).length <= 1) {
    document.body.classList.add('homepage');
  }
  
  // Add animation classes to feature cards
  document.querySelectorAll('.feature-card').forEach((card, index) => {
    card.style.animationDelay = `${index * 0.1}s`;
    card.classList.add('animate-in');
  });
});
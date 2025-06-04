(() => {
  // <stdin>
  document.addEventListener("DOMContentLoaded", function() {
    const toggle = document.querySelector("#theme-toggle");
    if (toggle) {
      const theme = localStorage.getItem("theme") || "light";
      if (theme === "dark") {
        applyDarkTheme();
      }
      toggle.addEventListener("click", function() {
        const isDark = document.body.style.backgroundColor === "rgb(0, 0, 0)" || document.body.style.backgroundColor === "#000" || document.body.style.backgroundColor === "#000000";
        if (isDark) {
          applyLightTheme();
        } else {
          applyDarkTheme();
        }
      });
    }
    function applyDarkTheme() {
      document.body.style.backgroundColor = "#000";
      document.body.style.color = "#fff";
      const style = document.createElement("style");
      style.id = "dark-theme-links";
      style.textContent = `
            a { color: #60a5fa !important; } /* lighter blue for dark mode */
            a:visited { color: #60a5fa !important; } /* same blue for visited links */
            blockquote { 
                color: #ccc !important; 
                border-left-color: #555 !important; 
            }
        `;
      document.head.appendChild(style);
      localStorage.setItem("theme", "dark");
    }
    function applyLightTheme() {
      document.body.style.backgroundColor = "";
      document.body.style.color = "";
      const darkStyle = document.getElementById("dark-theme-links");
      if (darkStyle) {
        darkStyle.remove();
      }
      localStorage.setItem("theme", "light");
    }
  });
})();

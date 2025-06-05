// raw HTML with minimal functionality
document.addEventListener('DOMContentLoaded', function () {
    // theme toggle
    const toggle = document.querySelector('#theme-toggle');
    if (toggle) {
        // Apply initial theme state (dark theme data-theme attribute should already be set by inline script)
        const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
        if (currentTheme === 'dark') {
            applyDarkThemeStyles();
        }

        toggle.addEventListener('click', function () {
            const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
            if (currentTheme === 'dark') {
                applyLightTheme();
            } else {
                applyDarkTheme();
            }
        });
    } function applyDarkTheme() {
        document.documentElement.setAttribute('data-theme', 'dark');
        localStorage.setItem('theme', 'dark');
        applyDarkThemeStyles();
    }

    function applyDarkThemeStyles() {
        // Create comprehensive dark theme styles for Hugo's inline syntax highlighting
        let existingStyle = document.getElementById('syntax-override-styles');
        if (existingStyle) {
            existingStyle.remove();
        }

        const style = document.createElement('style');
        style.id = 'syntax-override-styles';
        style.textContent = `
            /* Override Hugo's inline syntax highlighting styles in dark mode */
            [data-theme="dark"] .highlight span[style*="color:#1f2328"] { color: #e8e6e3 !important; } /* default text */
            [data-theme="dark"] .highlight span[style*="color:#cf222e"] { color: #ff7b72 !important; } /* keywords */
            [data-theme="dark"] .highlight span[style*="color:#0550ae"] { color: #79c0ff !important; } /* numbers/constants */
            [data-theme="dark"] .highlight span[style*="color:#6639ba"] { color: #d2a8ff !important; } /* functions */
            [data-theme="dark"] .highlight span[style*="color:#0a3069"] { color: #a5f3fc !important; } /* strings */
            [data-theme="dark"] .highlight span[style*="color:#57606a"] { color: #8b949e !important; } /* comments */
            [data-theme="dark"] .highlight span[style*="color:#6a737d"] { color: #8b949e !important; } /* comments alt */
            [data-theme="dark"] .highlight span[style*="color:#24292e"] { color: #e8e6e3 !important; } /* default alt */
            [data-theme="dark"] .highlight span[style*="color:#7f7f7f"] { color: #7d8590 !important; } /* line numbers */
            [data-theme="dark"] .highlight span[style*="color:#900"] { color: #ff7b72 !important; } /* error colors */
            
            /* Override Hugo's background colors */
            [data-theme="dark"] .highlight pre[style*="background-color:#fff"] { 
                background-color: #161b22 !important; 
            }
            [data-theme="dark"] .highlight[style*="background-color:#fff"] { 
                background-color: #161b22 !important; 
            }
        `;
        document.head.appendChild(style);
    }

    function applyLightTheme() {
        document.documentElement.setAttribute('data-theme', 'light');
        localStorage.setItem('theme', 'light');

        // Remove syntax override styles
        const existingStyle = document.getElementById('syntax-override-styles');
        if (existingStyle) {
            existingStyle.remove();
        }
    }
});
// raw HTML with minimal functionality
document.addEventListener('DOMContentLoaded', function () {    // theme toggle
    const toggle = document.querySelector('#theme-toggle');
    if (toggle) {
        const theme = localStorage.getItem('theme') || 'light';
        if (theme === 'dark') {
            applyDarkTheme();
        }

        toggle.addEventListener('click', function () {
            const isDark = document.body.style.backgroundColor === 'rgb(0, 0, 0)';
            if (isDark) {
                applyLightTheme();
            } else {
                applyDarkTheme();
            }
        });
    }

    function applyDarkTheme() {
        document.body.style.backgroundColor = '#000';
        document.body.style.color = '#fff';
        // Set link colors for dark mode
        const style = document.createElement('style');
        style.id = 'dark-theme-links';
        style.textContent = `
            a { color: #60a5fa !important; } /* lighter blue for dark mode */
            a:visited { color: #60a5fa !important; } /* same blue for visited links */
        `;
        document.head.appendChild(style);
        localStorage.setItem('theme', 'dark');
    }

    function applyLightTheme() {
        document.body.style.backgroundColor = '';
        document.body.style.color = '';
        // Remove dark theme link styles
        const darkStyle = document.getElementById('dark-theme-links');
        if (darkStyle) {
            darkStyle.remove();
        }
        localStorage.setItem('theme', 'light');
    }// math rendering - ensure KaTeX is loaded
    function renderMath() {
        if (typeof renderMathInElement !== 'undefined') {
            renderMathInElement(document.body, {
                delimiters: [
                    { left: '$$', right: '$$', display: true },
                    { left: '$', right: '$', display: false },
                    { left: '\\(', right: '\\)', display: false },
                    { left: '\\[', right: '\\]', display: true }
                ],
                throwOnError: false,
                macros: {
                    "\\RR": "\\mathbb{R}",
                    "\\NN": "\\mathbb{N}",
                    "\\ZZ": "\\mathbb{Z}",
                    "\\QQ": "\\mathbb{Q}",
                    "\\CC": "\\mathbb{C}"
                }
            });
        } else {
            // Retry after a delay if KaTeX not yet loaded
            setTimeout(renderMath, 100);
        }
    }    // Wait for KaTeX to load before rendering
    if (document.readyState === 'loading') {
        setTimeout(renderMath, 200);
    } else {
        renderMath();
    }    // Load more functionality for homepage sections
    function setupLoadMore(sectionName) {
        const loadMoreBtn = document.getElementById(sectionName + '-load-more');
        const items = document.querySelectorAll('.' + sectionName.replace('s', '') + '-item');

        if (loadMoreBtn && items.length > 3) {
            let expanded = false;

            loadMoreBtn.addEventListener('click', function () {
                if (!expanded) {
                    // Show all items
                    Array.from(items).forEach((item, index) => {
                        if (index >= 3) {
                            item.style.display = 'block';
                        }
                    });
                    loadMoreBtn.textContent = '[show less]';
                    expanded = true;
                } else {
                    // Hide items beyond first 3
                    Array.from(items).forEach((item, index) => {
                        if (index >= 3) {
                            item.style.display = 'none';
                        }
                    });
                    loadMoreBtn.textContent = '[load more]';
                    expanded = false;
                }
            });
        }
    }

    // Setup load more for each section
    setupLoadMore('blog');
    setupLoadMore('projects');
    setupLoadMore('publications');
});
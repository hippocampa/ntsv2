// raw HTML with minimal functionality
document.addEventListener('DOMContentLoaded', function () {
    // theme toggle
    const toggle = document.querySelector('#theme-toggle');
    if (toggle) {
        const theme = localStorage.getItem('theme') || 'light';
        if (theme === 'dark') {
            document.body.style.backgroundColor = '#000';
            document.body.style.color = '#fff';
        }

        toggle.addEventListener('click', function () {
            const isDark = document.body.style.backgroundColor === 'rgb(0, 0, 0)';
            if (isDark) {
                document.body.style.backgroundColor = '';
                document.body.style.color = '';
                localStorage.setItem('theme', 'light');
            } else {
                document.body.style.backgroundColor = '#000';
                document.body.style.color = '#fff';
                localStorage.setItem('theme', 'dark');
            }
        });
    }

    // math rendering - ensure KaTeX is loaded
    function renderMath() {
        if (typeof renderMathInElement !== 'undefined') {
            renderMathInElement(document.body, {
                delimiters: [
                    { left: '$$', right: '$$', display: true },
                    { left: '$', right: '$', display: false }
                ],
                throwOnError: false
            });
        } else {
            // Retry after a delay if KaTeX not yet loaded
            setTimeout(renderMath, 100);
        }
    }

    renderMath();
});
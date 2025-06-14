(()=>{document.addEventListener("DOMContentLoaded",function(){let o=document.querySelector("#theme-toggle");o&&((document.documentElement.getAttribute("data-theme")||"light")==="dark"&&a(),o.addEventListener("click",function(){(document.documentElement.getAttribute("data-theme")||"light")==="dark"?n():r()}));function r(){document.documentElement.setAttribute("data-theme","dark"),localStorage.setItem("theme","dark"),a()}function a(){let t=document.getElementById("syntax-override-styles");t&&t.remove();let e=document.createElement("style");e.id="syntax-override-styles",e.textContent=`
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
        `,document.head.appendChild(e)}function n(){document.documentElement.setAttribute("data-theme","light"),localStorage.setItem("theme","light");let t=document.getElementById("syntax-override-styles");t&&t.remove()}});})();

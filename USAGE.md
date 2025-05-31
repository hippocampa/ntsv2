# Site Usage Guide

## Getting Started

Your academic site is now ready! Here's how to add content and customize it further.

## Adding New Content

### Publications

To add a new publication:

```bash
hugo new publications/your-paper-title.md
```

Fill in the front matter:
```yaml
---
title: "Your Paper Title"
authors: ["Your Name", "Co-Author Name"]
year: 2024
venue: "Journal/Conference Name"
volume: "123"
pages: "45-67"
abstract: "Brief description of your research..."
doi: "10.1000/example.doi"
arxiv: "2024.12345"
github: "https://github.com/yourhandle/repo"
tags: ["machine learning", "optimization"]
date: 2024-05-31
draft: false
---
```

### Projects

To add a new project:

```bash
hugo new projects/project-name.md
```

Example front matter:
```yaml
---
title: "Project Name"
date: 2024-05-31
status: "Active Development"
technologies: ["Python", "JavaScript", "Hugo"]
github: "https://github.com/yourhandle/project"
demo: "https://project-demo.com"
tags: ["web development", "open source"]
draft: false
---
```

### Blog Posts

To add a new blog post:

```bash
hugo new blog/post-title.md
```

Example front matter:
```yaml
---
title: "Understanding Machine Learning Basics"
date: 2024-05-31
readingTime: 10
categories: ["Machine Learning", "Tutorial"]
tags: ["ml", "python", "tutorial"]
draft: false
---
```

## Mathematical Notation

Use standard LaTeX syntax:

- Inline math: `$f(x) = x^2$`
- Display math: `$$\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}$$`
- Aligned equations:
```latex
$$\begin{align}
f(x) &= ax^2 + bx + c \\
f'(x) &= 2ax + b
\end{align}$$
```

## Code Highlighting

Supported languages include Python, JavaScript, Go, Rust, C++, and more:

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

## Customization

### Site Settings

Edit `hugo.toml` to customize:

```toml
baseURL = 'https://yourdomain.com'
title = 'Your Site Name'

[params]
  description = "Your site description"
  author = "Your Name"
```

### Colors and Styling

Modify `themes/ntstheme/assets/css/main.css` to change colors:

```css
:root {
  --primary-color: #2980b9;
  --accent-color: #3498db;
  --text-color: #333;
}
```

### Navigation

Update menus in `hugo.toml`:

```toml
[menus]
  [[menus.main]]
    name = 'Custom Page'
    url = '/custom'
    weight = 60
```

## Building for Production

To build the site for deployment:

```bash
hugo --minify
```

Files will be generated in the `public/` directory.

## Deployment Options

### GitHub Pages
1. Push to GitHub repository
2. Enable Pages in repository settings
3. Set source to GitHub Actions
4. Use Hugo deployment action

### Netlify
1. Connect repository to Netlify
2. Set build command: `hugo --minify`
3. Set publish directory: `public`

### Vercel
1. Import repository to Vercel
2. Framework will be auto-detected
3. Deploy automatically

## SEO Optimization

The theme includes:
- Meta descriptions from content summaries
- Open Graph tags for social sharing
- Twitter Card support
- Canonical URLs
- Structured data for publications

## Performance Features

- Minimal JavaScript (only for math rendering)
- Optimized CSS with critical path optimization
- Responsive images
- Fast KaTeX rendering
- Clean, semantic HTML

## Best Practices

1. **Content Organization**: Use clear, descriptive filenames
2. **SEO**: Write meaningful summaries and descriptions
3. **Math**: Test mathematical notation before publishing
4. **Images**: Optimize images before adding to content
5. **Links**: Use relative links for internal content

## Troubleshooting

### Math Not Rendering
- Check JavaScript console for errors
- Ensure proper LaTeX syntax
- Verify KaTeX CDN is accessible

### Styling Issues
- Clear browser cache
- Check CSS compilation
- Verify Hugo extended version

### Build Errors
- Check front matter syntax
- Verify file encoding (UTF-8)
- Ensure all required fields are present

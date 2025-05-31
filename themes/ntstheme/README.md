# NTS Academic Theme

A minimalistic, fast-loading academic theme for Hugo designed specifically for researchers, academics, and technical professionals.

## Features

- **Minimalistic Design**: Clean, distraction-free layout focusing on content
- **Academic Focus**: Built-in support for publications, projects, and research notes  
- **Mathematical Notation**: Full LaTeX support using KaTeX
- **Fast Loading**: Optimized for performance with minimal JavaScript
- **SEO Friendly**: Comprehensive meta tags and structured data
- **Responsive**: Mobile-first design that works on all devices
- **Dark Mode**: Automatic dark mode support based on user preference
- **Redistributable**: MIT licensed for easy sharing and modification

## Sections

The theme provides specialized layouts for:

1. **Home**: Site introduction with recent updates
2. **Publications**: Academic papers with metadata (authors, year, venue, abstract)
3. **Projects**: Technical projects with GitHub/GitLab integration
4. **Blog**: Technical writing with math and code support
5. **About**: Personal/professional information

## Requirements

- Hugo Extended v0.116.0 or later
- Modern web browser with JavaScript enabled (for math rendering)

## Installation

1. Create a new Hugo site or navigate to existing site:
```bash
hugo new site my-academic-site
cd my-academic-site
```

2. Clone the theme:
```bash
git clone https://github.com/yourhandle/nts-academic-theme themes/ntstheme
```

3. Update your `hugo.toml`:
```toml
baseURL = 'https://yoursite.com'
languageCode = 'en-us'
title = 'Your Site Title'
theme = 'ntstheme'

[params]
  description = "Your site description"
  author = "Your Name"
```

## Content Structure

### Publications

Create publication files in `content/publications/`:

```yaml
---
title: "Your Paper Title"
authors: ["Your Name", "Co-Author"]
year: 2024
venue: "Conference/Journal Name"
abstract: "Paper abstract..."
doi: "10.1000/example"
---
```

### Projects

Create project files in `content/projects/`:

```yaml
---
title: "Project Name"
status: "Active Development"
technologies: ["Python", "JavaScript"]
github: "https://github.com/user/repo"
---
```

### Blog Posts

Create blog posts in `content/blog/`:

```yaml
---
title: "Post Title"
date: 2024-01-01
categories: ["Category"]
tags: ["tag1", "tag2"]
---
```

## Mathematical Notation

The theme supports LaTeX math notation via KaTeX:

- Inline math: `$E = mc^2$`
- Display math: `$$\int_a^b f(x)dx$$`

## Customization

### Colors

Modify colors in `assets/css/main.css`:

```css
:root {
  --primary-color: #2980b9;
  --text-color: #333;
  --background-color: #fdfdfd;
}
```

### Fonts

The theme uses Georgia for body text and Helvetica for headings. Modify in CSS:

```css
body {
  font-family: 'Georgia', 'Times New Roman', serif;
}

h1, h2, h3, h4, h5, h6 {
  font-family: 'Helvetica Neue', 'Arial', sans-serif;
}
```

## Performance

The theme is optimized for fast loading:

- Minimal JavaScript (only for math rendering)
- Optimized CSS with responsive design
- No external dependencies except KaTeX CDN
- Semantic HTML for better SEO

## Browser Support

- Chrome/Chromium 88+
- Firefox 85+
- Safari 14+
- Edge 88+

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

For issues, questions, or suggestions:

- Create an issue on GitHub
- Check existing documentation
- Review sample content for examples

## Credits

- Built with [Hugo](https://gohugo.io/)
- Math rendering by [KaTeX](https://katex.org/)
- Inspired by academic publishing standards

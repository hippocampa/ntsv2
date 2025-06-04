---
title: "Sample Post to Showcase Load More Functionality"
date: 2024-12-20T00:00:00Z
readingTime: 5
categories: ["Web Development", "Hugo"]
tags: ["hugo", "javascript", "ui", "demo"]
---

## Introduction

This is a sample blog post created specifically to demonstrate the "load more" functionality on the homepage. When you have more than 3 posts in a section, a clickable "[load more]" button will appear next to the section heading.

## How It Works

The load more functionality:

1. **Initial Display**: Shows only the first 3 items in each section (blog, projects, publications)
2. **Load More Button**: Appears when there are 4+ items in a section
3. **Toggle Behavior**: 
   - Click "[load more]" → Shows all items, button changes to "[show less]"
   - Click "[show less]" → Hides items beyond the first 3, button reverts to "[load more]"

## Mathematical Notation Example

Since this site supports KaTeX, here's a sample equation to test the math rendering:

The quadratic formula: $$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$

And some inline math: The famous equation $E = mc^2$ shows the mass-energy equivalence.

## Code Example

Here's a simple JavaScript function that demonstrates the toggle functionality:

```javascript
function toggleVisibility(items, threshold) {
    items.forEach((item, index) => {
        if (index >= threshold) {
            item.style.display = item.style.display === 'none' ? 'block' : 'none';
        }
    });
}
```

## Conclusion

This post serves as the 4th blog entry, which should trigger the appearance of the "[load more]" button on the homepage. Navigate back to the home page to see the functionality in action!

The raw HTML aesthetic is preserved while adding modern interactive features that enhance the user experience without compromising the minimalist design philosophy.

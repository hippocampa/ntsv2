// raw HTML with minimal functionality
document.addEventListener("DOMContentLoaded", function () {
	// theme toggle functionality disabled
	console.log("Theme toggle disabled");

	// Add image captions based on alt text
	addImageCaptions();
});

function addImageCaptions() {
	// Find all images in the content that have alt text
	const images = document.querySelectorAll("img[alt]");

	images.forEach(function (img) {
		const altText = img.getAttribute("alt");

		// Skip if alt text is empty, already has a caption, or is in a figure element
		if (
			!altText ||
			altText.trim() === "" ||
			img.parentElement.classList.contains("image-container") ||
			img.closest("figure")
		) {
			return;
		}

		// Skip if this appears to be a decorative image (common patterns)
		const decorativePatterns = /^(icon|logo|decoration|spacer)$/i;
		if (decorativePatterns.test(altText.trim())) {
			return;
		}

		// Create container div
		const container = document.createElement("div");
		container.className = "image-container";

		// Create caption element
		const caption = document.createElement("div");
		caption.className = "image-caption";
		caption.textContent = altText;

		// Insert container before the image
		img.parentNode.insertBefore(container, img);

		// Move image into container
		container.appendChild(img);

		// Add caption after image
		container.appendChild(caption);
	});
}

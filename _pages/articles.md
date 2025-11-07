---
permalink: /articles/
title: Articles
---

<!-- pages/projects.md -->
<div class="articles">
  <!-- Display categorized projects -->
  {% for category in page.display_categories %}
  <a id="{{ category }}" href=".#{{ category }}">
    <h2 class="category">{{ category }}</h2>
  </a>
  {% assign categorized_projects = site.articles %}
  {% assign sorted_projects = categorized_projects %}
  <!-- Generate cards for each project -->
  <div class="row row-cols-1 row-cols-md-3">
    {% for project in sorted_projects %}
      {% include articles.liquid %}
    {% endfor %}
  </div>
  {% endfor %}


<!-- Display projects without categories -->

{% assign sorted_projects = site.articles %}

  <!-- Generate cards for each project -->

  <div class="row row-cols-1 row-cols-md-3">
    {% for project in sorted_projects %}
      {% include articles.liquid %}
    {% endfor %}
  </div>
</div>
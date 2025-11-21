---
permalink: /articles/
title: Articles
layout: archive
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
  <div style="display: flex; flex-wrap: wrap; flex-direction: row; justify-content: space-between; align-items: center;">
    {% for project in sorted_projects %}
      {% include articles.liquid %}
    {% endfor %}
  </div>
  {% endfor %}


<!-- Display projects without categories -->

{% assign sorted_projects = site.articles %}

  <!-- Generate cards for each project -->

  <div style="display: flex; flex-wrap: wrap; flex-direction: row; justify-content: space-between; align-items: center;">
    {% for project in sorted_projects %}
      {% include articles.liquid %}
    {% endfor %}
  </div>
</div>
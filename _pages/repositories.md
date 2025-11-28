---
permalink: /repositories/
title: repositories
---

{% if site.data.repositories.github_users %}

## GitHub users

<div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 1rem;">
  {% for user in site.data.repositories.github_users %}
    {% include repository/repo_user.liquid username=user %}
  {% endfor %}
</div>

---

{% endif %}

{% if site.data.repositories.github_repos %}

## GitHub Repositories

<div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 1rem;">
  {% for repo in site.data.repositories.github_repos %}
    {% include repository/repo.liquid repository=repo %}
  {% endfor %}
</div>
{% endif %}
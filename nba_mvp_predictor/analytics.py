import os


def get_google_analytic_code():
    tag = get_google_analytics_tag()
    return f"""
  <!-- Global site tag (gtag.js) - Google Analytics -->
  <script async src="https://www.googletagmanager.com/gtag/js?id={tag}"></script>
  <script>
    window.dataLayer = window.dataLayer || [];
    function gtag(){{dataLayer.push(arguments);}}
    gtag('js', new Date());

    gtag('config', '{tag}');
  </script>
  """


def get_google_analytics_tag():
    tag = os.environ["GOOGLE_ANALYTICS_TAG"]
    return tag

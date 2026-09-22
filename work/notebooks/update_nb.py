import json

out_text = """Total valid rows: 28795
Sample row (transposed):
content_id                 content_304f48230142
client_id                     client_f369cb89fc
search_volume                                10
competition                                0.67
competition_level                          HIGH
cpc                                        2.05
content_type                    keyword article
main_intent                       transactional
word_count                                 3221
char_count                                20457
provider_used                                  
model_used                     gemini-2.5-flash
impressions_90d                            3803
clicks_90d                                   29
pageviews_90d                                22
sessions_90d                                 17
users_90d                                    16
engaged_sessions_90d                          1
ai_sessions_90d                               0
scroll_events_90d                             1
days_with_impressions                        88
days_with_sessions                           13
impressions_last_30d                        578
clicks_last_30d                               2
sessions_last_30d                             2
impressions_prev_30d                        987
clicks_prev_30d                              13
sessions_prev_30d                             9
content_age_days                            187
age_tier                                181-365
age_tier_order                                5
days_since_last_update                       20
freshness_tier                             0-30
word_count_tier                       2000-3500
char_count_tier                     15000-25000
ctr                                        0.76
avg_position                               10.6
engagement_rate                            5.88
scroll_rate                                4.55
ai_traffic_pct                                0
impression_tier                            good
position_tier                          striking
trend_direction                            down
trend_pct                                 -41.4
Name: 0, dtype: object
"""

nb_path = "w02_ml_task_framing.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

nb["cells"][8]["outputs"] = [{
    "name": "stdout",
    "output_type": "stream",
    "text": [line + "\n" for line in out_text.split("\n")]
}]
nb["cells"][8]["execution_count"] = 1

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

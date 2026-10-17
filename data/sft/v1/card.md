# Dataset v1: tau-telecom-sft

Built 2026-10-17T12:01:23Z at code `937eb98` from 912 conversations in 2 source(s): **518 kept**, 394 dropped. 74 of the 74 training tasks have at least one row.

|  |  |
|---|---|
| train.jsonl sha256 | `75718a646d5d2d85da1b2a9f42adb3b4f317f2f08614a36f4c07d95d6ee54745` |
| split | v3, sha256 `45d481019e630fba8a5a1e0b55f3fc789ba3118858300e8688c37b6c2fc4f90a` |
| tau2 pin | `2174a603f6d014ef94473ffa95957f6ce27100db` |
| system prompt sha256 | `b76554cc96138af12ae5a6a053b213a08148a4edb6e2de9d19cd63eb7ac5ddac` |
| tools sha256 | `66b1aa33c7b193fedd548535867cef95cd7612fd5758f992c79e2feb2877a165` |
| token counter | Qwen/Qwen3-4B-Instruct-2507 tokenizer and chat template; limit 24,576 |

## Sources

| id | source | file | bytes | sha256 | conversations |
|---|---|---|---|---|---|
| S1 | board/glm-5 | glm-5_enabled_telecom_gpt-5.2_4trials.json | 56,311,944 | `18e41808e95cb4fb…` | 456 |
| S2 | board/qwen3.5-397b | qwen3.5-397b-a17b_enabled_telecom_gpt-5.2_4trials.json | 71,968,908 | `4038e532bef560a3…` | 456 |

## Filters, in order

Each drop is charged to the first rule it failed.

| rule | keep a conversation only if | in | dropped | kept |
|---|---|---|---|---|
| F1 | tau2 scored it a pass (reward 1) | 912 | 70 | 842 |
| F2 | its task is one of split v3's 74 training tasks | 842 | 298 | 544 |
| F3 | it ended normally: the customer stopped, or the agent transferred when the task expects it | 544 | 1 | 543 |
| F4 | every agent tool call names one of the 13 agent tools | 543 | 7 | 536 |
| F5 | no agent tool result is an error | 536 | 4 | 532 |
| F6 | its agent turns are at most 1.5 x the median of its task's passes | 532 | 6 | 526 |
| F7 | no reply holds a tool call written as text | 526 | 0 | 526 |
| F8 | not a test twin: no test task's exact set of root issues | 526 | 0 | 526 |
| F9 | at most 3 conversations per task with the same agent tool calls | 526 | 8 | 518 |
| F10 | it fits the length limit once rendered | 518 | 0 | 518 |

### Why they were dropped

| rule: reason | conversations |
|---|---|
| F1: reward 0.0 | 18 |
| F1: reward None | 52 |
| F2: a test task | 298 |
| F3: transferred to a human when the task does not expect it | 1 |
| F4: called can_send_mms | 1 |
| F4: called check_apn_settings | 1 |
| F4: called check_network_mode_preference | 1 |
| F4: called check_sim_status | 1 |
| F4: called grant_app_permission | 2 |
| F4: called run_speed_test | 1 |
| F5: 1 agent tool error(s) | 4 |
| F6: 10 agent turns, over 1.5 x the task median 6.5 | 2 |
| F6: 12 agent turns, over 1.5 x the task median 7 | 2 |
| F6: 18 agent turns, over 1.5 x the task median 11 | 1 |
| F6: 25 agent turns, over 1.5 x the task median 15.5 | 1 |
| F9: copy 4 of one tool path | 6 |
| F9: copy 5 of one tool path | 2 |

## What was kept

### By symptom

| symptom | rows |
|---|---|
| mms_issue | 212 |
| mobile_data_issue | 165 |
| service_issue | 141 |

### By root issue

| root issue | rows |
|---|---|
| airplane_mode_on | 313 |
| bad_network_preference | 253 |
| bad_vpn | 59 |
| bad_wifi_calling | 92 |
| break_apn_mms_setting | 80 |
| break_apn_settings | 81 |
| break_app_both_permissions | 59 |
| break_app_sms_permission | 35 |
| break_app_storage_permission | 52 |
| contract_end_suspension | 25 |
| data_mode_off | 176 |
| data_saver_mode_on | 79 |
| data_usage_exceeded | 225 |
| lock_sim_card_pin | 96 |
| overdue_bill_suspension | 73 |
| unseat_sim_card | 176 |
| user_abroad_roaming_disabled_off | 100 |
| user_abroad_roaming_disabled_on | 89 |
| user_abroad_roaming_enabled_off | 101 |

### By source

| source | rows |
|---|---|
| S1 board/glm-5 | 245 |
| S2 board/qwen3.5-397b | 273 |

### By task

| task | rows |
|---|---|
| `[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Hard]` | 5 |
| `[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]` | 5 |
| `[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_storage_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_off[PERSONA:Easy]` | 6 |
| `[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_storage_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Easy]` | 5 |
| `[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_storage_permission|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_off[PERSONA:Easy]` | 5 |
| `[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_off[PERSONA:None]` | 6 |
| `[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_app_storage_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]` | 7 |
| `[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_app_storage_permission|unseat_sim_card[PERSONA:Hard]` | 7 |
| `[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Easy]` | 7 |
| `[mms_issue]airplane_mode_on|bad_network_preference|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|user_abroad_roaming_enabled_off[PERSONA:None]` | 4 |
| `[mms_issue]airplane_mode_on|bad_network_preference|break_apn_mms_setting|break_app_storage_permission[PERSONA:None]` | 8 |
| `[mms_issue]airplane_mode_on|bad_network_preference|break_apn_mms_setting|data_mode_off|unseat_sim_card|user_abroad_roaming_disabled_off[PERSONA:Easy]` | 7 |
| `[mms_issue]airplane_mode_on|bad_network_preference|break_apn_mms_setting|data_usage_exceeded[PERSONA:Hard]` | 7 |
| `[mms_issue]airplane_mode_on|bad_network_preference|break_app_both_permissions|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Hard]` | 6 |
| `[mms_issue]airplane_mode_on|bad_network_preference|break_app_storage_permission|data_mode_off|user_abroad_roaming_enabled_off[PERSONA:Easy]` | 7 |
| `[mms_issue]airplane_mode_on|bad_network_preference|data_usage_exceeded[PERSONA:None]` | 8 |
| `[mms_issue]airplane_mode_on|bad_wifi_calling|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:None]` | 4 |
| `[mms_issue]airplane_mode_on|bad_wifi_calling|user_abroad_roaming_enabled_off[PERSONA:Easy]` | 8 |
| `[mms_issue]airplane_mode_on|break_app_both_permissions[PERSONA:Hard]` | 7 |
| `[mms_issue]airplane_mode_on|break_app_both_permissions|data_usage_exceeded|user_abroad_roaming_disabled_off[PERSONA:None]` | 7 |
| `[mms_issue]bad_network_preference|bad_wifi_calling|break_app_sms_permission|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]` | 7 |
| `[mms_issue]bad_network_preference|break_app_both_permissions[PERSONA:Easy]` | 7 |
| `[mms_issue]bad_network_preference|break_app_sms_permission|data_mode_off|data_usage_exceeded|user_abroad_roaming_enabled_off[PERSONA:None]` | 7 |
| `[mms_issue]bad_network_preference|break_app_sms_permission|user_abroad_roaming_disabled_on[PERSONA:Hard]` | 7 |
| `[mms_issue]bad_network_preference|data_mode_off|user_abroad_roaming_disabled_on[PERSONA:None]` | 7 |
| `[mms_issue]bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|user_abroad_roaming_disabled_off[PERSONA:None]` | 6 |
| `[mms_issue]bad_wifi_calling|break_apn_mms_setting|data_mode_off|data_usage_exceeded|unseat_sim_card[PERSONA:None]` | 7 |
| `[mms_issue]bad_wifi_calling|break_apn_mms_setting|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Easy]` | 7 |
| `[mms_issue]break_apn_mms_setting|data_mode_off|user_abroad_roaming_disabled_on[PERSONA:Hard]` | 8 |
| `[mms_issue]break_app_both_permissions|data_usage_exceeded[PERSONA:Hard]` | 8 |
| `[mms_issue]break_app_sms_permission|data_usage_exceeded|user_abroad_roaming_disabled_on[PERSONA:None]` | 8 |
| `[mms_issue]break_app_storage_permission|data_usage_exceeded[PERSONA:Easy]` | 7 |
| `[mobile_data_issue]airplane_mode_on|bad_network_preference|bad_vpn|data_mode_off|data_saver_mode_on|data_usage_exceeded|user_abroad_roaming_disabled_off[PERSONA:Hard]` | 7 |
| `[mobile_data_issue]airplane_mode_on|bad_network_preference|bad_vpn|data_mode_off|data_saver_mode_on|user_abroad_roaming_disabled_off[PERSONA:Hard]` | 8 |
| `[mobile_data_issue]airplane_mode_on|bad_network_preference|bad_vpn|data_saver_mode_on|data_usage_exceeded|user_abroad_roaming_enabled_off[PERSONA:None]` | 6 |
| `[mobile_data_issue]airplane_mode_on|bad_network_preference|bad_vpn|data_saver_mode_on|user_abroad_roaming_disabled_on[PERSONA:Easy]` | 7 |
| `[mobile_data_issue]airplane_mode_on|bad_network_preference|data_mode_off|data_saver_mode_on[PERSONA:Hard]` | 6 |
| `[mobile_data_issue]airplane_mode_on|bad_network_preference|data_mode_off|data_saver_mode_on|data_usage_exceeded|user_abroad_roaming_disabled_on[PERSONA:None]` | 6 |
| `[mobile_data_issue]airplane_mode_on|bad_network_preference|data_mode_off|data_usage_exceeded|user_abroad_roaming_disabled_on[PERSONA:None]` | 8 |
| `[mobile_data_issue]airplane_mode_on|bad_network_preference|data_mode_off|user_abroad_roaming_disabled_on[PERSONA:Hard]` | 8 |
| `[mobile_data_issue]airplane_mode_on|bad_network_preference|data_saver_mode_on|data_usage_exceeded|user_abroad_roaming_disabled_off[PERSONA:None]` | 8 |
| `[mobile_data_issue]airplane_mode_on|bad_network_preference|data_usage_exceeded|user_abroad_roaming_disabled_off[PERSONA:Easy]` | 8 |
| `[mobile_data_issue]airplane_mode_on|bad_network_preference|user_abroad_roaming_enabled_off[PERSONA:Easy]` | 8 |
| `[mobile_data_issue]airplane_mode_on|user_abroad_roaming_enabled_off[PERSONA:None]` | 8 |
| `[mobile_data_issue]bad_network_preference|bad_vpn|data_mode_off|data_saver_mode_on|data_usage_exceeded|user_abroad_roaming_enabled_off[PERSONA:Easy]` | 8 |
| `[mobile_data_issue]bad_network_preference|bad_vpn|data_usage_exceeded|user_abroad_roaming_disabled_off[PERSONA:Easy]` | 8 |
| `[mobile_data_issue]bad_network_preference|bad_vpn|user_abroad_roaming_disabled_on[PERSONA:None]` | 7 |
| `[mobile_data_issue]bad_network_preference|data_mode_off|data_saver_mode_on|data_usage_exceeded|user_abroad_roaming_disabled_off[PERSONA:Easy]` | 8 |
| `[mobile_data_issue]bad_network_preference|data_saver_mode_on|data_usage_exceeded[PERSONA:Hard]` | 7 |
| `[mobile_data_issue]bad_vpn|data_mode_off|data_usage_exceeded|user_abroad_roaming_disabled_off[PERSONA:None]` | 8 |
| `[mobile_data_issue]data_mode_off|data_usage_exceeded[PERSONA:None]` | 8 |
| `[mobile_data_issue]data_mode_off|data_usage_exceeded|user_abroad_roaming_disabled_off[PERSONA:Hard]` | 8 |
| `[mobile_data_issue]data_saver_mode_on|user_abroad_roaming_enabled_off[PERSONA:Easy]` | 8 |
| `[mobile_data_issue]data_usage_exceeded|user_abroad_roaming_enabled_off[PERSONA:Easy]` | 7 |
| `[service_issue]airplane_mode_on|break_apn_settings|contract_end_suspension|unseat_sim_card[PERSONA:Easy]` | 7 |
| `[service_issue]airplane_mode_on|break_apn_settings|lock_sim_card_pin[PERSONA:None]` | 8 |
| `[service_issue]airplane_mode_on|break_apn_settings|lock_sim_card_pin|overdue_bill_suspension[PERSONA:Hard]` | 8 |
| `[service_issue]airplane_mode_on|break_apn_settings|lock_sim_card_pin|overdue_bill_suspension|unseat_sim_card[PERSONA:None]` | 7 |
| `[service_issue]airplane_mode_on|break_apn_settings|lock_sim_card_pin|unseat_sim_card[PERSONA:None]` | 7 |
| `[service_issue]airplane_mode_on|break_apn_settings|overdue_bill_suspension[PERSONA:None]` | 4 |
| `[service_issue]airplane_mode_on|break_apn_settings|overdue_bill_suspension|unseat_sim_card[PERSONA:None]` | 6 |
| `[service_issue]airplane_mode_on|break_apn_settings|unseat_sim_card[PERSONA:None]` | 6 |
| `[service_issue]airplane_mode_on|lock_sim_card_pin[PERSONA:Easy]` | 8 |
| `[service_issue]airplane_mode_on|lock_sim_card_pin|overdue_bill_suspension[PERSONA:Easy]` | 8 |
| `[service_issue]airplane_mode_on|lock_sim_card_pin|overdue_bill_suspension|unseat_sim_card[PERSONA:Easy]` | 8 |
| `[service_issue]airplane_mode_on|lock_sim_card_pin|unseat_sim_card[PERSONA:Hard]` | 8 |
| `[service_issue]airplane_mode_on|overdue_bill_suspension|unseat_sim_card[PERSONA:Easy]` | 8 |
| `[service_issue]airplane_mode_on|unseat_sim_card[PERSONA:None]` | 6 |
| `[service_issue]break_apn_settings|contract_end_suspension|lock_sim_card_pin[PERSONA:Hard]` | 6 |
| `[service_issue]break_apn_settings|contract_end_suspension|lock_sim_card_pin|unseat_sim_card[PERSONA:Hard]` | 6 |
| `[service_issue]break_apn_settings|lock_sim_card_pin|overdue_bill_suspension|unseat_sim_card[PERSONA:Easy]` | 8 |
| `[service_issue]break_apn_settings|overdue_bill_suspension|unseat_sim_card[PERSONA:Hard]` | 8 |
| `[service_issue]contract_end_suspension|lock_sim_card_pin[PERSONA:Hard]` | 6 |
| `[service_issue]lock_sim_card_pin|overdue_bill_suspension[PERSONA:Easy]` | 8 |

## Length of the kept rows (tokens)

Counted with the Qwen/Qwen3-4B-Instruct-2507 tokenizer and chat template. The system prompt and the 13 tools alone take 7,134 tokens. Kept rows have a median of 14 agent turns (max 32).

| min | median | p90 | p99 | max |
|---|---|---|---|---|
| 8205 | 10054 | 11111 | 12437 | 16869 |

### Histogram (1,024-token bins)

```
 8,192 ############## 79
 9,216 ######################################## 223
10,240 ################################ 177
11,264 ###### 32
12,288 # 4
13,312  0
14,336 # 2
15,360  0
16,384 # 1
```

### What a limit would cut, of the 518 rows that reached F10

| limit | rows over it |
|---|---|
| 8,192 | 518 |
| 12,288 | 7 |
| 16,384 | 1 |
| 20,480 | 0 |
| 24,576 | 0 |

## Source checks (§4: five things M1 checks after downloading)

|  | S1 GLM-5 | S2 Qwen3.5-397B-A17B |
|---|---|---|
| agent model, as tau2 recorded it | `openai/glm-5-fp8` | `openai/qwen3.5-397b` |
| agent sampling, as tau2 recorded it | `{'temperature': 0.0}` | `{'temperature': 1.0, 'top_p': 0.95}` |
| customer model | `gpt-5.2` | `gpt-5.2` |
| customer settings | `{'reasoning_effort': 'low'}` | `{'reasoning_effort': 'low'}` |
| tau2 commit the board ran | `unknown` | `unknown` |
| trials | `4` | `4` |
| customer phone-tool results in the file (removed from every row) | `3639` | `4464` |
| agent tool results | `2743` | `2760` |
| assistant turns with reasoning in raw_data (dropped) | `4636` | `6032` |
| assistant turns with both text and tool calls (kept as is) | `955` | `249` |
| system prompt stored in the file | `False` | `False` |
| the file's policy is the pin's, byte for byte | `True` | `True` |
| licence | `none found beside the files; tau2-bench itself is MIT` | `none found beside the files; tau2-bench itself is MIT` |

## Licence

tau2-bench is MIT. The published board conversations carry no separate licence. Customer turns are another model's output and are context only, with no loss on them. No Claude output is in this dataset.

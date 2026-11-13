# 使用者 review 3：緊急對應模式「找到的不是那顆球」（資料：outputs/lab/live/emerg_e1.json；14:43 已被你吸收的指示＋使用者附圖）

## 使用者看到的問題
球（sphere）滑脫後，系統「找到了」一個東西並抓起來，但**抓的是旁邊的 cylinder**，放進籃子；原本的球被留在原地或滾走。使用者要的是：知道沒抓到 → 立刻提高一點 → 在抓取點附近**找到那顆球** → 抓它。

## 你目前的盲點（請對照資料確認，不同意就寫證據）
**事實（emerg_e1.json，三組 emergency=True 全部一樣）：**
| case | emergency「found」位置 | 真正的 cylinder_d50_h40 位置 | 實際入籃 | 被滑走的 sphere_60 最後在哪 |
|---|---|---|---|---|
| roll_slow | (−0.671, −0.002) | (−0.671, −0.005) | cylinder + sphere_40 | 掉出桌外 (−1.16, 0.35) |
| roll_fast | (−0.676, −0.002) | 同上 | cylinder + sphere_40 | 掉出桌外 (−1.75, 2.8) |
| roll_x | (−0.674, −0.003) | 同上 | cylinder + sphere_40 | **仍在桌上 (−0.63, −0.08)，離 cylinder 約 85 mm** |
- 「found」的位置距離 cylinder **不到 5 mm**。三次都是把鄰居的 cylinder 當成滑走的球，並且 `live_pick.py` 把追蹤 7 號的中心**直接改成那個位置**（見 `em_["xy"]` → `t_["center"]`），然後抓了它。
- `ok=2` 看起來比基準 `ok=1` 好，但**被算進去的是 cylinder**。`in_basket`（只算 sphere）兩種模式都是 1。**你的指標把「抓到任何東西」當成功，所以沒發現。**
- `roll_x` 球還在桌上、可回收，卻選了更近的 cylinder：搜尋沒有「身分」的概念。
- 另外 `roll_slow / roll_fast` 的球根本掉出桌外，這兩組本來就**不可回收**，正確行為是回報「球離開桌面」並去抓下一顆，而不是找一個替代品。

**這個問題的根因（請用自己的話說一次）：** 「class-free blob + 大小約等於 w」只能回答「附近有一個差不多大的東西」，不能回答「這是不是那顆球」。尺寸容許（你用的範圍把 d50 當成 59 mm）太鬆，而且沒有排除「本來就在那裡的其他物體」。

## 要的修正：先確認身分，找不到就承認找不到
1. **排除法（最重要）：** 掃描時追蹤到的其他物體（鄰居）有已知位置。緊急搜尋的候選 blob 若與任何**其他追蹤**重合（< 約 15–20 mm），就**不是**滑走的球，直接排除。滑走的球只可能出現在「原本沒有東西」的位置，或原位置附近。
2. **身分驗證（兩個都要過）：** (a) 候選 blob 在 hover 靜止幀上做形狀擬合，必須是球（用既有 sphere 擬合與殘差門檻，不要只看外框大小）或 YOLO 類別為 sphere；(b) 寬度與原本 w 相差 ≤ 約 10%（不是 30%）。通過才算 `found`。
3. **不要改寫追蹤：** 沒通過就 `lost`（附原因：離桌／被排除／沒看到），**不可**把追蹤中心改成別的物體的位置。
4. **抓之前的最後確認：** 在 hover 靜止幀確認要抓的那個物體類別 = 目標類別（`detect_pick("sphere")` 就只能抓 sphere）；不符就中止，記為 `wrong_class_prevented`。
5. `lost` 之後才去抓下一顆 target。

## 測試與指標要改（沒改就看不出有沒有進步）
- **主指標：被滑走的那顆球（用真值名稱 `sphere_60`）最後是否入籃**；**輔助指標：`wrong_object_picked`（被移動的非目標物件數，用 batch2 已有的 `moved_other` 邏輯）**。`ok` 只當參考。
- 每個 case 的**預期結果**要事先寫好：
  - 球停在桌上、可回收 → 應入籃，`wrong_object_picked = 0`。
  - 球掉出桌外 → 應回報 `lost: off table`，`wrong_object_picked = 0`，並繼續抓下一顆。
- **一定要有「鄰居就在旁邊」的場景**（目前 seed 31 的 cylinder 剛好在旁邊，是很好的壓力測試，保留）。也要有鄰居不在旁邊的場景，確認沒有被過度排除。
- 每個 case ≥ 5 個不同樣本（換 seed／方向／滑動距離），不要重複相同條件。

## 順序（一次一件，每做完一個編號回報一次，≤10 行）
| # | 事項 | 完成標準 |
|---|---|---|
| 1 | 用文字回答：上面三組為什麼 `found` 的是 cylinder？你的「找到」是用什麼條件決定的？（引用 `emerg_e1.json` 與 `live_pick.py` 的位置） | ≤10 行，附行號 |
| 2 | 改指標：加入 `slipped_target_in_basket`、`wrong_object_picked`、每個 case 的預期結果。**用目前的程式先重算 e1 / e2**，應該看到 wrong_object_picked ≥ 1 | 一張表，e1 三組的 wrong_object_picked 數字 |
| 3 | 實作排除法＋身分驗證＋不改寫追蹤＋抓前類別確認（上面 1–5） | 同樣的 case：wrong_object_picked = 0；roll_x 的 `slipped_target_in_basket = True`；掉出桌外的 case 回報 `lost: off table` |
| 4 | 補到每個 case ≥ 5 個樣本，附 HTML 總表（含影片路徑） | `outputs/lab/live/react_report.html` |
| 5 | 之後才回頭處理：`removed` 時 refine 不去抓其他 target 的 bug、`target_gone` 的誤判（高度門檻、連續幀、半徑內搜尋） | 見 react_s2.json 對照 |

## 不要做
- 在 #2 做完、確認舊程式確實抓錯之前，不要先改行為。
- 不要把「抓到任何東西」當成成功。
- 不要用真值當控制輸入（真值只做驗收與診斷，包含 `truth_watch`）。
- 不要把緊急模式或 `visual_refine` 預設開啟，直到所有 case 的 `wrong_object_picked = 0` 且被滑走的球回收率不低於基準。
- README 更新與上傳 `hayashimatsu/arm-detect-and-pick`：使用者說 #5 驗證後再商量，**先不要做**。

// static/bus_monitor.js
// 每秒輪詢 /bus_vision/status，更新右側資訊面板

let isDetectionComplete = false; // 是否已完成辨識
let lastBestNumber = null; // 最後一次辨識成功的號碼

function formatTs(ts) {
  if (!ts) return "—";
  const d = new Date(ts * 1000);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleString();
}

async function fetchBusStatus() {
  // 如果已經完成辨識，不再更新
  if (isDetectionComplete) {
    return;
  }
  
  try {
    const resp = await fetch("/bus_vision/status");
    if (!resp.ok) return;
    const s = await resp.json();

    // 最有信心的公車號碼
    // 當收集到5個偵測結果時，強制從中選出最有信心的（即使信心度很低也要選）
    const bestNumEl = document.getElementById("best-bus-number");
    const bestConfEl = document.getElementById("best-conf");
    const detectionCount = s.detection_count || 0;
    const requiredCount = s.required_count || 5;
    const allDetections = s.all_detections || [];
    const hasEnough = detectionCount >= requiredCount;
    
    let bestNumber = s.best_bus_number;
    let bestConf = s.best_confidence;
    
    // 如果已經收集到5個但還沒有選出最有信心的，且信心度達到90%，才選一個
    if (hasEnough && allDetections.length >= 5 && !bestNumber) {
      const bestItem = allDetections.reduce((max, item) => 
        item.confidence > max.confidence ? item : max
      );
      // 只有信心度達到90%才顯示
      if (bestItem.confidence >= 0.90) {
        bestNumber = bestItem.bus_number;
        bestConf = bestItem.confidence;
      }
    }
    
    // 檢查是否達到90%信心度
    // 只有在已經開始辨識（有影像流顯示）且達到90%信心度時，才顯示成功訊息
    const streamImg = document.getElementById("bus-stream");
    const isStreamVisible = streamImg && streamImg.style.display !== "none" && streamImg.src && streamImg.src.includes("/bus_vision/stream");
    
    // 驗證號碼是否有效（不能是"0"或空字串）
    const isValidNumber = bestNumber && bestNumber !== "0" && bestNumber.trim() !== "" && bestNumber.match(/^\d{1,4}$/);
    
    if (isValidNumber && bestConf >= 0.90 && isStreamVisible) {
      // 辨識成功，立即停止AI模型並顯示成功訊息
      // 只有在影像流正在顯示時（表示已經開始辨識），才認為是真正的成功
      if (!isDetectionComplete) {
        isDetectionComplete = true;
        lastBestNumber = bestNumber;
        
        // 關閉影像流（停止手機攝像頭）
        if (streamImg) {
          streamImg.style.display = "none";
          streamImg.src = ""; // 停止影像流
        }
        
        // 隱藏GPS提示
        const gpsMsg = document.getElementById("gps-message");
        if (gpsMsg) gpsMsg.style.display = "none";
        
        // 顯示成功訊息
        const successMsg = document.getElementById("success-message");
        const successBusNum = document.getElementById("success-bus-number");
        if (successMsg && successBusNum) {
          successBusNum.innerHTML = `目標公車 <span style="color:#f97316;font-weight:700;">${bestNumber}</span> 號即將進站`;
          successMsg.style.display = "block";
        }
        
        // 3秒後切換到"公車行駛中"訊息
        setTimeout(() => {
          if (successMsg) successMsg.style.display = "none";
          const busRidingMsg = document.getElementById("bus-riding-message");
          if (busRidingMsg) {
            busRidingMsg.style.display = "block";
          }
          console.log("[bus_monitor] 已切換到公車行駛中訊息");
        }, 3000);
        
        // 停止輪詢（不再更新）
        console.log("[bus_monitor] 辨識成功，AI模型已關閉，停止更新，顯示成功訊息");
      }
      
      bestNumEl.textContent = bestNumber;
      bestNumEl.style.color = "#f97316";
      if (typeof bestConf === "number") {
        bestConfEl.textContent = (bestConf * 100).toFixed(1) + "%";
      } else {
        bestConfEl.textContent = "—";
      }
    } else {
      // 如果還沒有開始辨識（影像流未顯示），不顯示成功訊息
      // 也不要在橘色大字上顯示任何內容
      if (!isStreamVisible) {
        bestNumEl.textContent = "—";
        bestNumEl.style.color = "#64748b";
        bestConfEl.textContent = "—";
      } else {
        // 已經開始辨識但還沒達到90%，顯示當前最佳結果
        if (bestNumber) {
          bestNumEl.textContent = bestNumber;
          bestNumEl.style.color = "#f97316";
          if (typeof bestConf === "number") {
            bestConfEl.textContent = (bestConf * 100).toFixed(1) + "%";
          } else {
            bestConfEl.textContent = "—";
          }
        } else {
          bestNumEl.textContent = "—";
          bestNumEl.style.color = "#64748b";
          bestConfEl.textContent = "—";
        }
      }
    }

    // 所有偵測到的號碼（一行一行顯示，後面括弧辨識度）
    // 顯示所有5個原始偵測結果
    const allDetectionsEl = document.getElementById("all-detections");
    
    console.log("[bus_monitor] 偵測結果:", {
      all_detections: s.all_detections,
      detection_count: detectionCount,
      required_count: requiredCount
    });
    
    // 清空現有內容
    allDetectionsEl.innerHTML = "";
    
    // 顯示所有偵測結果（一行一行往下排列）
    if (s.all_detections && Array.isArray(s.all_detections) && s.all_detections.length > 0) {
      // 顯示所有偵測結果（一行一行往下排列，格式：號碼 (辨識度%)）
      s.all_detections.forEach((det, idx) => {
        const div = document.createElement("div");
        div.style.cssText = "padding:10px 14px;background:#1e293b;border-radius:8px;border:1px solid #334155;margin-bottom:6px;";
        
        // 格式：號碼 (辨識度%)
        const busNum = det.bus_number || "未知";
        const confidence = typeof det.confidence === "number" ? det.confidence : 0;
        const text = `${busNum} (${(confidence * 100).toFixed(1)}%)`;
        
        div.textContent = text;
        div.style.cssText += "font-size:1rem;color:#e5e7eb;font-weight:500;line-height:1.5;";
        
        allDetectionsEl.appendChild(div);
      });
      
      // 如果還沒收集到5個，顯示進度
      if (detectionCount < requiredCount) {
        const progressDiv = document.createElement("div");
        progressDiv.style.cssText = "padding:8px 12px;background:#0f172a;border-radius:8px;border:1px solid #475569;margin-top:8px;text-align:center;";
        progressDiv.textContent = `進度: ${detectionCount}/${requiredCount}`;
        progressDiv.style.cssText += "font-size:0.9rem;color:#94a3b8;font-weight:600;";
        allDetectionsEl.appendChild(progressDiv);
      } else {
        // 已經收集到5個，顯示完成訊息
        const completeDiv = document.createElement("div");
        completeDiv.style.cssText = "padding:8px 12px;background:#0f172a;border-radius:8px;border:1px solid #22c55e;margin-top:8px;text-align:center;";
        completeDiv.textContent = `✓ 已完成收集 (${detectionCount}/${requiredCount})`;
        completeDiv.style.cssText += "font-size:0.9rem;color:#22c55e;font-weight:600;";
        allDetectionsEl.appendChild(completeDiv);
      }
    } else {
      // 沒有偵測結果，顯示進度
      const emptyDiv = document.createElement("div");
      emptyDiv.style.cssText = "padding:12px;text-align:center;";
      emptyDiv.innerHTML = `<div style="color:#64748b;font-size:0.9rem;margin-bottom:8px;">尚未偵測到任何號碼</div><div style="color:#475569;font-size:0.85rem;">進度: ${detectionCount}/${requiredCount}</div>`;
      allDetectionsEl.appendChild(emptyDiv);
    }

    // 最近一次辨識
    const numEl = document.getElementById("bus-number");
    const timeEl = document.getElementById("bus-time");
    const rawEl = document.getElementById("raw-text");

    numEl.textContent = s.bus_number || "尚未辨識到車號";
    timeEl.textContent = formatTs(s.last_seen_ts);
    rawEl.textContent = s.raw_text || "—";
  } catch (e) {
    console.error("fetchBusStatus error:", e);
  }
}

function initBusMonitor() {
  // 初始化時，確保所有元素都是隱藏的
  const streamImg = document.getElementById("bus-stream");
  const successMsg = document.getElementById("success-message");
  const gpsMsg = document.getElementById("gps-message");
  
  if (streamImg) streamImg.style.display = "none";
  if (successMsg) successMsg.style.display = "none";
  if (gpsMsg) gpsMsg.style.display = "none";
  
  // 重置狀態
  isDetectionComplete = false;
  lastBestNumber = null;
  
  fetchBusStatus();
  // 只有在未完成辨識時才繼續輪詢
  const pollInterval = setInterval(() => {
    if (!isDetectionComplete) {
      fetchBusStatus();
    } else {
      clearInterval(pollInterval);
    }
  }, 1000);
}

// 接收來自 monitor.js 的 GPS 位置提示和辨識成功訊息
window.addEventListener("message", (event) => {
  if (event.data && event.data.type === "gps_message") {
    const gpsMsg = document.getElementById("gps-message");
    const gpsText = document.getElementById("gps-text");
    const streamImg = document.getElementById("bus-stream");
    const successMsg = document.getElementById("success-message");
    
    // 只有在距離>20公尺時才顯示GPS提示
    // 識別模式下（距離<=20公尺）不顯示GPS提示文字
    if (gpsMsg && gpsText && !isDetectionComplete) {
      // 檢查是否已經開始辨識（影像流是否顯示）
      const isStreamVisible = streamImg && streamImg.style.display !== "none" && streamImg.src && streamImg.src.includes("/bus_vision/stream");
      
      // 如果還沒開始辨識，才顯示GPS提示
      if (!isStreamVisible) {
        gpsText.textContent = event.data.message;
        gpsMsg.style.display = "block";
        if (streamImg) streamImg.style.display = "none";
        if (successMsg) successMsg.style.display = "none";
      } else {
        // 已經開始辨識，隱藏GPS提示
        if (gpsMsg) gpsMsg.style.display = "none";
      }
    }
  } else if (event.data && event.data.type === "start_detection") {
    // 開始辨識時，隱藏GPS提示，只顯示影像流（手機攝像頭）和AI運算結果
    const gpsMsg = document.getElementById("gps-message");
    const streamImg = document.getElementById("bus-stream");
    const successMsg = document.getElementById("success-message");
    const busRidingMsg = document.getElementById("bus-riding-message");
    
    // 隱藏所有文字提示，只顯示影像流
    if (gpsMsg) gpsMsg.style.display = "none";
    if (successMsg) successMsg.style.display = "none";
    if (busRidingMsg) busRidingMsg.style.display = "none";
    
    if (streamImg && !isDetectionComplete) {
      console.log("[bus_monitor] 收到 start_detection 訊息，準備顯示影像流");
      
      // 先檢查後端是否已經啟動，如果沒有則先啟動
      async function checkAndStartStream() {
        try {
          // 先確保後端已經啟動 bus_vision（使用手機模式）
          console.log("[bus_monitor] 確保後端 bus_vision 已啟動...");
          const startResp = await fetch("/bus_vision/start?use_mobile=true", {
            method: "POST"
          });
          
          if (startResp.ok) {
            const startResult = await startResp.json();
            console.log("[bus_monitor] 後端 bus_vision 啟動結果:", startResult);
          } else {
            console.warn("[bus_monitor] 後端 bus_vision 啟動失敗，狀態碼:", startResp.status);
            // 即使啟動失敗，也嘗試顯示影像流（可能已經啟動了）
          }
          
          // 等待一小段時間確保後端已準備好
          await new Promise(resolve => setTimeout(resolve, 500));
          
          // 檢查後端狀態
          const statusResp = await fetch("/bus_vision/status");
          if (statusResp.ok) {
            const status = await statusResp.json();
            console.log("[bus_monitor] 後端狀態:", status);
          }
          
          // 設置影像流URL（添加時間戳避免快取，強制刷新）
          // 使用隨機參數確保每次都是新的連接，避免瀏覽器快取
          const streamUrl = "/bus_vision/stream?t=" + Date.now() + "&r=" + Math.random();
          if (streamImg) {
            // 先停止舊的流
            streamImg.src = "";
            // 強制重新載入
            setTimeout(() => {
              if (streamImg && !isDetectionComplete) {
                streamImg.src = streamUrl;
                streamImg.style.display = "block";
                // 強制刷新，避免快取
                streamImg.crossOrigin = "anonymous";
                console.log("[bus_monitor] 已設置影像流URL:", streamUrl);
              }
            }, 100);
          }
          
          // 添加錯誤處理和自動重連
          let reconnectAttempts = 0;
          const maxReconnectAttempts = 5;
          
          streamImg.onerror = function() {
            reconnectAttempts++;
            console.warn(`[bus_monitor] 影像流載入失敗，重試中... (${reconnectAttempts}/${maxReconnectAttempts})`);
            if (reconnectAttempts < maxReconnectAttempts && streamImg && !isDetectionComplete) {
              // 等待1秒後重試，使用新的時間戳
              setTimeout(() => {
                if (streamImg && !isDetectionComplete) {
                  streamImg.src = "/bus_vision/stream?t=" + Date.now() + "&r=" + Math.random();
                }
              }, 1000);
            }
          };
          
          // 當影像流成功載入時，重置重連計數
          streamImg.onload = function() {
            if (reconnectAttempts > 0) {
              console.log("[bus_monitor] 影像流已成功重新連接");
              reconnectAttempts = 0;
            }
          };
        } catch (err) {
          console.error("[bus_monitor] 檢查後端狀態失敗:", err);
          // 即使檢查失敗，也嘗試顯示影像流
          if (streamImg) {
            streamImg.src = "/bus_vision/stream?" + Date.now();
            streamImg.style.display = "block";
          }
        }
      }
      
      checkAndStartStream();
      console.log("[bus_monitor] 已開啟手機攝像頭影像辨識模式，顯示後置鏡頭畫面");
    }
  } else if (event.data && event.data.type === "detection_success") {
    // 辨識成功，顯示成功訊息
    isDetectionComplete = true;
    const streamImg = document.getElementById("bus-stream");
    const successMsg = document.getElementById("success-message");
    const successBusNum = document.getElementById("success-bus-number");
    const gpsMsg = document.getElementById("gps-message");
    const busRidingMsg = document.getElementById("bus-riding-message");
    
    if (streamImg) {
      streamImg.style.display = "none";
      streamImg.src = ""; // 停止影像流
    }
    if (gpsMsg) gpsMsg.style.display = "none";
    if (busRidingMsg) busRidingMsg.style.display = "none";
    if (successMsg && successBusNum) {
      successBusNum.innerHTML = `目標公車 <span style="color:#f97316;font-weight:700;">${event.data.busNumber}</span> 號即將進站`;
      successMsg.style.display = "block";
    }
    
    // 3秒後切換到"公車行駛中"訊息
    setTimeout(() => {
      if (successMsg) successMsg.style.display = "none";
      if (busRidingMsg) {
        busRidingMsg.style.display = "block";
      }
      console.log("[bus_monitor] 已切換到公車行駛中訊息");
    }, 3000);
    
    console.log("[bus_monitor] 辨識成功，已關閉影像流，顯示成功訊息");
  }
});

document.addEventListener("DOMContentLoaded", initBusMonitor);

// static/monitor.js
// 後台地圖監看：顯示 Google overview polyline + 使用者即時位置

let map;
let routePolyline = null;
let userMarker = null;
let startMarker = null;
let endMarker = null;
let lastPolylineStr = null;
let deviceId = null;
let pollTimer = null;
let busMonitorWindow = null;
let lastBusStepIndex = null;
let busMonitorOpened = false; // 標記是否已打開 bus_monitor 視窗

// 和 app.js 共用 device_id，但優先使用 URL 參數或手動輸入
function getDeviceId() {
  // 優先使用 URL 參數
  const urlParams = new URLSearchParams(window.location.search);
  const urlDeviceId = urlParams.get("device_id");
  if (urlDeviceId) {
    return urlDeviceId;
  }
  
  // 其次使用 localStorage
  const KEY = "blindnav_device_id";
  let id = window.localStorage.getItem(KEY);
  if (!id) {
    const rand = Math.random().toString(36).slice(2, 10);
    id = "device_" + rand;
    window.localStorage.setItem(KEY, id);
  }
  return id;
}

function setDeviceId(newId) {
  if (!newId || !newId.trim()) {
    console.error("[monitor] setDeviceId: 無效的 device_id");
    return false;
  }
  
  const trimmedId = newId.trim();
  console.log("[monitor] 設定 device_id:", trimmedId);
  
  deviceId = trimmedId;
  const KEY = "blindnav_device_id";
  window.localStorage.setItem(KEY, deviceId);
  
  const devSpan = document.getElementById("device-id-display");
  if (devSpan) {
    devSpan.textContent = deviceId;
    console.log("[monitor] 已更新顯示的 device_id");
  }
  
  // 停止舊的輪詢
  if (pollTimer !== null) {
    clearInterval(pollTimer);
    pollTimer = null;
    console.log("[monitor] 已停止舊的輪詢");
  }
  
  // 立即測試一次連接
  console.log("[monitor] 開始測試連接...");
  fetchSnapshotOnce().then(() => {
    console.log("[monitor] 測試連接完成");
  }).catch((err) => {
    console.error("[monitor] 測試連接失敗:", err);
  });
  
  // 重新開始輪詢
  startPolling();
  console.log("[monitor] 已重新開始輪詢");
  
  return true;
}

async function fetchSnapshotOnce() {
  if (!deviceId) {
    console.warn("[monitor] fetchSnapshotOnce: device_id 為空");
    return;
  }

  try {
    const url = `/device_state/${encodeURIComponent(deviceId)}`;
    console.log("[monitor] 請求 device_state:", url);
    
    const resp = await fetch(url);
    if (!resp.ok) {
      const errorText = await resp.text();
      console.warn("[monitor] device_state 錯誤:", resp.status, errorText);
      
      // 如果是 404，表示裝置不存在或尚未註冊
      if (resp.status === 404) {
        const devSpan = document.getElementById("device-id-display");
        if (devSpan) {
          devSpan.textContent = `${deviceId} (未找到 - 請確認手機端已註冊)`;
          devSpan.style.color = "#ef4444"; // 紅色提示
        }
        // 清空所有顯示
        clearAllDisplay();
        return;
      }
      return;
    }
    
    const snap = await resp.json();
    console.log("[monitor] device_state 回應:", snap);
    
    // 更新顯示的 device_id（確保同步）
    const devSpan = document.getElementById("device-id-display");
    if (devSpan) {
      devSpan.textContent = deviceId;
      devSpan.style.color = ""; // 恢復正常顏色
    }
    
    // 檢查是否有路線資料
    if (!snap.route) {
      console.log("[monitor] 裝置已連接，但尚未設定路線");
      // 顯示連接成功但無路線的狀態
      updateUIFromSnapshot(snap, true); // 傳入標記表示已連接但無路線
    } else {
      updateUIFromSnapshot(snap, false);
    }
  } catch (e) {
    console.error("[monitor] fetchSnapshotOnce 異常:", e);
    // 網路錯誤時也清空顯示
    const devSpan = document.getElementById("device-id-display");
    if (devSpan) {
      devSpan.textContent = `${deviceId} (連接錯誤)`;
      devSpan.style.color = "#ef4444";
    }
  }
}

// 清空所有顯示
function clearAllDisplay() {
  const lastSeenEl = document.getElementById("last-seen");
  const stepTextEl = document.getElementById("step-text");
  const etaEl = document.getElementById("eta");
  const stepListEl = document.getElementById("step-list");
  const startAddrEl = document.getElementById("start-addr");
  const endAddrEl = document.getElementById("end-addr");
  const reqDepTimeEl = document.getElementById("req-dep-time");
  
  if (lastSeenEl) lastSeenEl.textContent = "—";
  if (stepTextEl) stepTextEl.textContent = "裝置未找到或尚未註冊";
  if (etaEl) etaEl.textContent = "—";
  if (stepListEl) stepListEl.textContent = "";
  if (startAddrEl) startAddrEl.textContent = "—";
  if (endAddrEl) endAddrEl.textContent = "—";
  if (reqDepTimeEl) reqDepTimeEl.textContent = "—";
}

function formatTime(t) {
  if (!t) return "—";
  try {
    const d = new Date(t);
    if (Number.isNaN(d.getTime())) return String(t);
    return d.toLocaleString();
  } catch {
    return String(t);
  }
}

function checkAndOpenBusMonitor(snap) {
  // 檢查是否需要觸發公車監視（視窗已經打開，只需要觸發視訊）
  if (!snap || !snap.route || !snap.route.steps) {
    return; // 沒有路線，但保持視窗打開
  }
  
  const steps = snap.route.steps;
  const currentStepIndex = typeof snap.last_step_index === "number" 
    ? Math.min(Math.max(snap.last_step_index, 0), steps.length - 1)
    : 0;
  
  // 確保 bus_monitor 視窗已打開
  if (!busMonitorWindow || busMonitorWindow.closed) {
    busMonitorWindow = window.open("/bus_monitor", "_blank");
    busMonitorOpened = true;
    console.log("[monitor] 已重新打開公車監視畫面");
  }
  
  // 檢查是否需要觸發視訊（接近公車站時）
  // 尋找下一個 bus_ride 步驟
  let nextBusStep = null;
  let nextBusStepIndex = null;
  for (let i = currentStepIndex; i < steps.length; i++) {
    if (steps[i].mode === "bus_ride") {
      nextBusStep = steps[i];
      nextBusStepIndex = i;
      break;
    }
  }
  
  // 如果下一步是公車，計算距離並顯示GPS提示
  if (nextBusStep && nextBusStepIndex !== null) {
    let distance = null;
    let gpsMessage = null;
    
    // 公車站的位置應該是當前步驟（walk）的結束位置，而不是公車步驟的目標位置
    // 找到當前步驟（在公車步驟之前的 walk 步驟）
    let currentStep = null;
    if (currentStepIndex >= 0 && currentStepIndex < steps.length) {
      currentStep = steps[currentStepIndex];
    }
    
    // 如果當前步驟沒有位置，嘗試使用前一個步驟的結束位置
    let busStopLat = null;
    let busStopLng = null;
    
    if (currentStep && currentStep.target_lat && currentStep.target_lng) {
      // 使用當前步驟的結束位置作為公車站位置
      busStopLat = currentStep.target_lat;
      busStopLng = currentStep.target_lng;
    } else if (nextBusStepIndex > 0) {
      // 如果當前步驟沒有位置，使用前一個步驟的結束位置
      const prevStep = steps[nextBusStepIndex - 1];
      if (prevStep && prevStep.target_lat && prevStep.target_lng) {
        busStopLat = prevStep.target_lat;
        busStopLng = prevStep.target_lng;
      }
    }
    
    // 計算距離（使用模擬的實際位置和公車站位置）
    if (typeof snap.last_lat === "number" && typeof snap.last_lng === "number" && 
        busStopLat !== null && busStopLng !== null) {
      distance = haversineDistance(
        snap.last_lat,
        snap.last_lng,
        busStopLat,
        busStopLng
      );
      
      console.log(`[monitor] 距離公車站: ${Math.round(distance)}公尺 (位置: ${snap.last_lat}, ${snap.last_lng} -> ${busStopLat}, ${busStopLng})`);
      
      // 根據距離顯示不同的GPS提示
      // 當距離<=20公尺時，不顯示GPS提示，直接進入辨識模式
      if (distance <= 20) {
        // 距離<=20公尺，立即觸發辨識模式，不顯示GPS提示文字
        console.log(`[monitor] 距離<=${Math.round(distance)}公尺，立即開啟辨識模式（從GPS訊息觸發）`);
        // 立即通知 bus_monitor 開始辨識（不顯示GPS提示文字）
        if (busMonitorWindow && !busMonitorWindow.closed) {
          busMonitorWindow.postMessage({
            type: "start_detection"
          }, "*");
        }
        // 同時觸發 shouldTrigger，確保後續邏輯也執行
        if (lastBusStepIndex !== nextBusStepIndex) {
          lastBusStepIndex = nextBusStepIndex;
        }
        // 不設置 gpsMessage，直接跳過GPS訊息發送
        gpsMessage = null;
      } else {
        // 距離>20公尺時，顯示GPS提示
        if (distance > 100) {
          gpsMessage = `再${Math.round(distance)}公尺到達公車站牌`;
        } else if (distance > 50) {
          gpsMessage = `再${Math.round(distance)}公尺到達公車站牌`;
        } else if (distance > 20) {
          gpsMessage = `再${Math.round(distance)}公尺到達公車站牌`;
        } else {
          gpsMessage = `再${Math.round(distance)}公尺到達公車站牌`;
        }
      }
      
      // 發送GPS提示到 bus_monitor 視窗（只有在距離>20公尺時才發送）
      if (busMonitorWindow && !busMonitorWindow.closed && gpsMessage) {
        busMonitorWindow.postMessage({
          type: "gps_message",
          message: gpsMessage
        }, "*");
      }
    }
    
    // 距離 100 公尺內就觸發視訊
    // 或者當模擬點已經到達或超過公車站步驟時，立即觸發
    // 或者當距離<5公尺（顯示"公車10秒內進站"）時，立即觸發
    let shouldTrigger = false;
    
    // 檢查模擬點是否已經到達公車站（通過步驟索引判斷）
    const isAtBusStop = currentStepIndex >= nextBusStepIndex - 1;
    
    if (distance !== null) {
      // 有距離資訊
      if (distance <= 100) {
        shouldTrigger = true;
      } else if (distance <= 20) {
        // 距離<=20公尺，立即觸發辨識模式（將臨界值從5公尺提高到20公尺）
        shouldTrigger = true;
        console.log(`[monitor] 距離<=${Math.round(distance)}公尺，立即開啟辨識模式`);
      } else if (isAtBusStop) {
        // 即使距離顯示還很遠，但如果模擬點已經到達公車站，也觸發
        console.log(`[monitor] 模擬點已到達公車站（步驟索引: ${currentStepIndex} >= ${nextBusStepIndex - 1}），立即觸發`);
        shouldTrigger = true;
      }
    } else if (isAtBusStop) {
      // 沒有距離資訊，但模擬點已經到達公車站，立即觸發
      console.log(`[monitor] 模擬點已到達公車站（步驟索引: ${currentStepIndex} >= ${nextBusStepIndex - 1}），立即觸發`);
      shouldTrigger = true;
    } else if (busStopLat === null || busStopLng === null) {
      // 完全沒有位置資訊，但下一步是公車，也觸發
      shouldTrigger = true;
    }
    
    if (shouldTrigger && lastBusStepIndex !== nextBusStepIndex) {
      lastBusStepIndex = nextBusStepIndex;
      console.log(`[monitor] 已觸發公車監視（接近公車站，距離: ${distance !== null ? Math.round(distance) + '公尺' : '未知'}）`);
      
      // 通知 bus_monitor 開始辨識（顯示手機攝像頭影像）
      // 這會隱藏GPS提示文字，顯示手機後置鏡頭畫面（有識別的框框）
      if (busMonitorWindow && !busMonitorWindow.closed) {
        busMonitorWindow.postMessage({
          type: "start_detection"
        }, "*");
      }
    }
    
    // 檢查是否需要關閉視訊（離開公車站或完成公車步驟）
    if (nextBusStepIndex === null || currentStepIndex > nextBusStepIndex) {
      // 已經過了公車步驟，關閉視訊
      fetch("/bus_vision/status")
        .then(resp => resp.ok ? resp.json() : null)
        .then(busStatus => {
          // 如果視訊還在運行，關閉它
          if (busStatus && busStatus.last_seen_ts) {
            fetch("/bus_vision/stop", { method: "POST" }).catch(console.error);
            console.log("[monitor] 已關閉公車監視視訊（離開公車站）");
          }
        })
        .catch(err => console.error("[monitor] 檢查公車狀態失敗:", err));
    }
  }
}

function haversineDistance(lat1, lng1, lat2, lng2) {
  const R = 6371000;
  const toRad = (d) => (d * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLng = toRad(lng2 - lng1);
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(toRad(lat1)) *
      Math.cos(toRad(lat2)) *
      Math.sin(dLng / 2) *
      Math.sin(dLng / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

function updateUIFromSnapshot(snap, isConnectedButNoRoute = false) {
  const lastSeenEl = document.getElementById("last-seen");
  const stepTextEl = document.getElementById("step-text");
  const etaEl = document.getElementById("eta");
  const stepListEl = document.getElementById("step-list");
  const startAddrEl = document.getElementById("start-addr");
  const endAddrEl = document.getElementById("end-addr");
  const reqDepTimeEl = document.getElementById("req-dep-time");

  if (!snap) return;

  // 更新最後更新時間
  if (lastSeenEl) {
    lastSeenEl.textContent = formatTime(snap.last_seen);
  }
  
  // 如果已連接但沒有路線，顯示特殊狀態
  if (isConnectedButNoRoute) {
    if (stepTextEl) {
      stepTextEl.textContent = "✓ 裝置已連接，等待手機端設定路線...";
      stepTextEl.style.color = "#22c55e"; // 綠色表示連接成功
    }
    if (etaEl) etaEl.textContent = "—";
    if (startAddrEl) startAddrEl.textContent = "—";
    if (endAddrEl) endAddrEl.textContent = "—";
    if (stepListEl) stepListEl.textContent = "請在手機端設定目的地並規劃路線";
    if (reqDepTimeEl) reqDepTimeEl.textContent = "—";
    return;
  }
  
  // 檢查是否需要打開公車監視
  checkAndOpenBusMonitor(snap);
  
  // 恢復正常顏色
  if (stepTextEl) stepTextEl.style.color = "";

  if (snap.route && snap.route.steps && snap.route.steps.length > 0) {
    const steps = snap.route.steps;
    const idx =
      typeof snap.last_step_index === "number"
        ? Math.min(Math.max(snap.last_step_index, 0), steps.length - 1)
        : 0;
    const curStep = steps[idx];

    if (stepTextEl) {
      stepTextEl.textContent = curStep
        ? `第 ${idx + 1} 步（${curStep.mode}）：${curStep.instruction || ""}`
        : "尚無步驟資料";
    }

    if (typeof snap.route.total_duration_min === "number" && etaEl) {
      const mins = Math.round(snap.route.total_duration_min);
      etaEl.textContent = `${mins} 分鐘`;
    } else if (etaEl) {
      etaEl.textContent = "—";
    }

    if (startAddrEl) startAddrEl.textContent = snap.route.start_address || "—";
    if (endAddrEl) endAddrEl.textContent = snap.route.end_address || "—";

    if (reqDepTimeEl) {
      if (snap.route.requested_departure_text) {
        reqDepTimeEl.textContent = snap.route.requested_departure_text;
      } else if (typeof snap.route.requested_departure_epoch === "number") {
        reqDepTimeEl.textContent = formatTime(
          snap.route.requested_departure_epoch * 1000
        );
      } else {
        reqDepTimeEl.textContent = "—";
      }
    }

    if (stepListEl) {
      const lines = steps.map((s) => {
        const base = `[#${s.index}] (${s.mode}) ${s.instruction || ""}`;
        if (s.mode === "bus_ride" && s.bus_number) {
          const dep = s.departure_stop ? `從「${s.departure_stop}」` : "";
          const arr = s.arrival_stop ? `到「${s.arrival_stop}」` : "";
          return `${base}\n    公車：${s.bus_number} ${dep}${arr}`;
        }
        return base;
      });
      stepListEl.textContent = lines.join("\n");
    }

    // overview polyline
    if (snap.route.overview_polyline && map && window.google) {
      const polyStr = snap.route.overview_polyline;
      if (polyStr !== lastPolylineStr) {
        lastPolylineStr = polyStr;

        if (routePolyline) {
          routePolyline.setMap(null);
          routePolyline = null;
        }

        const path = google.maps.geometry.encoding.decodePath(polyStr);
        routePolyline = new google.maps.Polyline({
          map,
          path,
        });

        const bounds = new google.maps.LatLngBounds();
        path.forEach((p) => bounds.extend(p));
        if (!bounds.isEmpty()) {
          map.fitBounds(bounds);
        }
      }
    }

    // 起點/終點 marker
    if (map && window.google) {
      if (
        typeof snap.route.start_lat === "number" &&
        typeof snap.route.start_lng === "number"
      ) {
        const spos = {
          lat: snap.route.start_lat,
          lng: snap.route.start_lng,
        };
        if (!startMarker) {
          startMarker = new google.maps.Marker({
            map,
            position: spos,
            label: "S",
            title: "起點",
          });
        } else {
          startMarker.setPosition(spos);
        }
      }

      if (
        typeof snap.route.end_lat === "number" &&
        typeof snap.route.end_lng === "number"
      ) {
        const epos = {
          lat: snap.route.end_lat,
          lng: snap.route.end_lng,
        };
        if (!endMarker) {
          endMarker = new google.maps.Marker({
            map,
            position: epos,
            label: "G",
            title: "終點",
          });
        } else {
          endMarker.setPosition(epos);
        }
      }
    }
  } else {
    if (stepTextEl) stepTextEl.textContent = "尚未取得該裝置的路線。";
    if (etaEl) etaEl.textContent = "—";
    if (startAddrEl) startAddrEl.textContent = "—";
    if (endAddrEl) endAddrEl.textContent = "—";
    if (stepListEl) stepListEl.textContent = "";
    if (reqDepTimeEl) reqDepTimeEl.textContent = "—";
  }

  // 使用者目前位置
  if (
    map &&
    typeof snap.last_lat === "number" &&
    typeof snap.last_lng === "number" &&
    window.google
  ) {
    const pos = { lat: snap.last_lat, lng: snap.last_lng };

    if (!userMarker) {
      userMarker = new google.maps.Marker({
        map,
        position: pos,
        title: "使用者目前位置",
      });
      map.setCenter(pos);
      if (!map.getZoom || map.getZoom() < 16) {
        map.setZoom(16);
      }
    } else {
      userMarker.setPosition(pos);
      map.panTo(pos);
    }
  }
}

function startPolling() {
  fetchSnapshotOnce();
  if (pollTimer !== null) {
    clearInterval(pollTimer);
  }
  // 0.3 秒輪詢一次，才看得出速度差異
  pollTimer = setInterval(fetchSnapshotOnce, 300);
}

function initMonitorMap() {
  // 先檢查 URL 參數
  const urlParams = new URLSearchParams(window.location.search);
  const urlDeviceId = urlParams.get("device_id");
  
  // 如果有 URL 參數，直接使用它（會自動設定）
  if (urlDeviceId) {
    console.log("[monitor] 從 URL 參數取得 device_id:", urlDeviceId);
    deviceId = urlDeviceId.trim();
    const KEY = "blindnav_device_id";
    window.localStorage.setItem(KEY, deviceId);
  } else {
    // 否則使用預設邏輯
  deviceId = getDeviceId();
  }
  
  const devSpan = document.getElementById("device-id-display");
  if (devSpan) devSpan.textContent = deviceId;
  console.log("[monitor] 初始 device_id:", deviceId);
  
  // 一開始就打開 bus_monitor 視窗（但不啟動視訊）
  if (!busMonitorOpened) {
    busMonitorWindow = window.open("/bus_monitor", "_blank");
    busMonitorOpened = true;
    console.log("[monitor] 已自動打開公車監視畫面（初始）");
  }
  
  // 設定 device_id 輸入框
  const deviceIdInput = document.getElementById("device-id-input");
  const setDeviceIdBtn = document.getElementById("set-device-id");
  
  if (deviceIdInput) {
    // 如果 URL 有參數，自動填入並設定
    if (urlDeviceId) {
      deviceIdInput.value = urlDeviceId;
      // 自動設定（不等待用戶點擊）
      console.log("[monitor] 自動設定 URL 參數中的 device_id");
      setDeviceId(urlDeviceId);
    }
    
    // Enter 鍵也可以設定
    deviceIdInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter") {
        setDeviceIdBtn?.click();
      }
    });
  }
  
  if (setDeviceIdBtn) {
    setDeviceIdBtn.addEventListener("click", () => {
      const input = deviceIdInput?.value?.trim();
      console.log("[monitor] 點擊設定按鈕，輸入值:", input);
      
      if (input) {
        if (setDeviceId(input)) {
          if (deviceIdInput) deviceIdInput.value = "";
          // 不顯示 alert，改用 console 和狀態更新
          console.log(`[monitor] 已設定裝置 ID：${deviceId}`);
          
          // 更新狀態提示（如果有的話）
          const statusMsg = `已設定裝置 ID：${deviceId}，正在同步...`;
          console.log("[monitor]", statusMsg);
          
          // 可以選擇顯示一個臨時提示
          const btnText = setDeviceIdBtn.textContent;
          setDeviceIdBtn.textContent = "已設定！";
          setDeviceIdBtn.style.background = "#22c55e";
          setTimeout(() => {
            setDeviceIdBtn.textContent = btnText;
            setDeviceIdBtn.style.background = "";
          }, 2000);
        } else {
          console.error("[monitor] 設定 device_id 失敗");
          alert("設定失敗，請檢查輸入的裝置 ID 格式");
        }
      } else {
        alert("請輸入有效的裝置 ID");
      }
    });
  }

  map = new google.maps.Map(document.getElementById("map"), {
    center: { lat: 23.7, lng: 120.9 },
    zoom: 8,
    mapTypeId: "roadmap",
  });

  const btn = document.getElementById("refresh-device");
  if (btn) {
    btn.addEventListener("click", fetchSnapshotOnce);
  }

  // 如果 URL 沒有參數，才開始輪詢（因為 setDeviceId 已經會開始輪詢了）
  if (!urlDeviceId) {
  startPolling();
  }
  // 如果有 URL 參數，setDeviceId 已經開始輪詢了，不需要重複
}

window.initMonitorMap = initMonitorMap;

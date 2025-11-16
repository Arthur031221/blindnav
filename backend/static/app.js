// static/app.js

// ==== 全域狀態 ====
let deviceId = null;
let currentRoute = null;
let currentStepIndex = 0;
let hasRoute = false;

let isEmergency = false;
let isSimulating = false;

let lastDistanceToTarget = null;
let spokenThresholdsPerStep = {};

let busVisionActive = false;
let busVisionSpoken = false;
let waitingForBusDetection = false; // <<< 新增
let mobileStream = null; // 手機視訊流
let mobileStreamInterval = null; // 發送視訊流的 interval



const DIST_THRESHOLDS = [100, 50, 20, 10, 5, 1];

const USE_FIXED_ORIGIN = true;
const FIXED_ORIGIN = {
  lat: 24.796123,
  lng: 120.9935,
};

let currentOrigin = null;
let currentSimLatLng = null;

let simSpeedFactor = 1.0;

// DOM
let destInput;
let departureDateInput;
let departureTimeInput;
let simSpeedInput;
let btnGoogleRoute;
let btnDemoRoute;
let btnSimulateRoute;
let btnEmergency;
let btnEnd;
let touchArea;
let statusLine;
let stepInfo;

// ==== 語音 ====

// 快取語音列表，避免每次都查詢
let cachedVoices = null;
let voicesReady = false;

// 初始化語音列表
function initVoices() {
  if (voicesReady) return;
  
  const voices = speechSynthesis.getVoices();
  if (voices.length > 0) {
    cachedVoices = voices;
    voicesReady = true;
    console.log("[TTS] 語音列表已載入，共", voices.length, "個語音");
  }
}

// 監聽語音列表載入事件
if (typeof speechSynthesis !== 'undefined') {
  speechSynthesis.onvoiceschanged = initVoices;
  // 立即嘗試載入（某些瀏覽器可能已經載入）
  initVoices();
}

function findChineseVoice() {
  if (!cachedVoices || cachedVoices.length === 0) {
    initVoices();
    if (!cachedVoices || cachedVoices.length === 0) {
      return null;
    }
  }
  
  // 優先尋找台灣中文語音
  let voice = cachedVoices.find(v => 
    v.lang === "zh-TW" || 
    v.lang === "zh-CN" ||
    v.lang.startsWith("zh")
  );
  
  // 如果找不到，尋找包含 "Chinese" 或 "Taiwan" 的語音
  if (!voice) {
    voice = cachedVoices.find(v => 
      v.name.toLowerCase().includes("chinese") ||
      v.name.toLowerCase().includes("taiwan") ||
      v.name.toLowerCase().includes("mandarin")
    );
  }
  
  return voice || null;
}

function shortenTextForTTS(text) {
  if (!text) return "";
  const maxLen = 40; // 約 5 秒內講完
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen) + "……";
}

function speak(rawText) {
  const text = shortenTextForTTS(rawText);
  if (!text) return;
  console.log("[TTS]", text);

  // 清空 queue 避免排太多句
  window.speechSynthesis.cancel();

  const utter = new SpeechSynthesisUtterance(text);
  
  // 設定語言
  utter.lang = "zh-TW";
  
  // 明確選擇中文語音
  const chineseVoice = findChineseVoice();
  if (chineseVoice) {
    utter.voice = chineseVoice;
    console.log("[TTS] 使用語音:", chineseVoice.name, chineseVoice.lang);
  } else {
    console.warn("[TTS] 找不到中文語音，使用預設語音");
  }
  
  utter.rate = 1.5; // 1.5倍語速
  speechSynthesis.speak(utter);
}

// ==== 小工具 ====

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function updateStatus(text) {
  if (statusLine) statusLine.textContent = text;
}

function updateStepInfo() {
  if (!stepInfo) return;
  if (!hasRoute || !currentRoute || !currentRoute.steps) {
    stepInfo.textContent = "";
    return;
  }
  const steps = currentRoute.steps;
  if (currentStepIndex < 0 || currentStepIndex >= steps.length) {
    stepInfo.textContent = "";
    return;
  }
  const s = steps[currentStepIndex];
  stepInfo.textContent = `第 ${currentStepIndex + 1} 步 / 共 ${
    steps.length
  } 步：(${s.mode}) ${s.instruction || ""}`;
}

// ==== device_id & 註冊 ====

function ensureDeviceId() {
  const KEY = "blindnav_device_id";
  let id = window.localStorage.getItem(KEY);
  if (!id) {
    const rand = Math.random().toString(36).slice(2, 10);
    id = "device_" + rand;
    window.localStorage.setItem(KEY, id);
  }
  deviceId = id;
  console.log("device_id =", deviceId);
}

async function registerDevice() {
  // 確保 device_id 已生成
  if (!deviceId) {
    ensureDeviceId();
  }
  
  // 更新顯示（確保顯示最新的 device_id）
  updateDeviceIdDisplay();

  try {
    const resp = await fetch("/register_device", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ device_id: deviceId }),
    });
    if (!resp.ok) {
      updateStatus("註冊裝置失敗。");
      console.error("register_device error:", resp.status, await resp.text());
      return;
    }
    const data = await resp.json();
    console.log("register_device response:", data);
    updateStatus("裝置已註冊，可以開始設定目的地。");
    
    // 註冊成功後再次更新顯示（確保同步）
    updateDeviceIdDisplay();
  } catch (e) {
    console.error("registerDevice exception:", e);
    updateStatus("註冊裝置時發生錯誤。");
  }
}

// ==== 與後端 API ====

async function sendLocationUpdate(lat, lng, stepIndex) {
  if (!deviceId) return;
  try {
    await fetch("/update_location", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        device_id: deviceId,
        lat,
        lng,
        step_index: typeof stepIndex === "number" ? stepIndex : currentStepIndex,
      }),
    });
  } catch (e) {
    console.error("sendLocationUpdate error:", e);
  }
}

async function requestDemoRoute() {
  if (!deviceId) {
    speak("裝置尚未註冊，請稍後再試。");
    return;
  }

  try {
    const resp = await fetch("/route", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ device_id: deviceId }),
    });

    if (!resp.ok) {
      speak("取得示範路線時發生錯誤。");
      console.error("route demo error:", resp.status, await resp.text());
      return;
    }

    const data = await resp.json();
    currentRoute = data.route;
    hasRoute = true;
    currentStepIndex = 0;
    spokenThresholdsPerStep = {};
    lastDistanceToTarget = null;

    console.log("Demo route:", currentRoute);
    updateStepInfo();

    if (currentRoute.summary) {
      speak(currentRoute.summary);
    } else {
      speak("已載入示範路線。");
    }
  } catch (e) {
    console.error("requestDemoRoute exception:", e);
    speak("取得示範路線時發生錯誤。");
  }
}

async function requestGoogleRoute(originLat, originLng, destinationText, depEpoch) {
  if (!deviceId) {
    speak("裝置尚未註冊，請稍後再試。");
    return null;
  }
  const payload = {
    device_id: deviceId,
    origin_lat: originLat,
    origin_lng: originLng,
    destination: destinationText,
  };
  if (typeof depEpoch === "number") {
    payload.departure_time = depEpoch;
  }

  try {
    const resp = await fetch("/route_google", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!resp.ok) {
      const text = await resp.text();
      console.error("route_google error:", resp.status, text);
      speak("取得 Google 路線時發生錯誤，請稍後再試。");
      return null;
    }

    const data = await resp.json();
    const route = data.route;
    console.log("Google route:", route);
    
    if (!route) {
      speak("無法規劃路線，請確認目的地是否正確。");
      return null;
    }
    
    return route;
  } catch (e) {
    console.error("requestGoogleRoute exception:", e);
    speak("取得 Google 路線時發生錯誤，請檢查網路連線。");
    return null;
  }
}

// ==== 距離提示 ====

function getNextBusStep(currIndex) {
  if (!currentRoute || !currentRoute.steps) return null;
  const steps = currentRoute.steps;
  for (let i = currIndex + 1; i < steps.length; i++) {
    if (steps[i].mode === "bus_ride") return steps[i];
  }
  return null;
}

function getNextBusArrivalPhrase(currIndex) {
  const busStep = getNextBusStep(currIndex);
  if (!busStep || !busStep.bus_number) return null;

  const busNum = busStep.bus_number;
  const ts = busStep.bus_departure_timestamp;
  if (!ts) return `接下來要搭乘公車 ${busNum}。`;

  const now = Math.floor(Date.now() / 1000);
  let diffMin = Math.round((ts - now) / 60);
  if (diffMin <= 0) {
    return `接下來要搭乘公車 ${busNum}，公車隨時可能進站。`;
  }
  return `接下來要搭乘公車 ${busNum}，大約 ${diffMin} 分鐘後到站。`;
}

function handleDistanceHints(distance) {
  if (!hasRoute || !currentRoute || !currentRoute.steps) return;
  const steps = currentRoute.steps;
  if (currentStepIndex < 0 || currentStepIndex >= steps.length) return;

  const currStep = steps[currentStepIndex];
  if (currStep.mode !== "walk") {
    lastDistanceToTarget = distance;
    return;
  }

  if (lastDistanceToTarget == null) {
    lastDistanceToTarget = distance;
    return;
  }

  const spokenForThisStep =
    spokenThresholdsPerStep[currStep.index] ||
    (spokenThresholdsPerStep[currStep.index] = {});

  for (const T of DIST_THRESHOLDS) {
    if (distance <= T && lastDistanceToTarget > T && !spokenForThisStep[T]) {
      spokenForThisStep[T] = true;

      const baseMsg = `距離下一個動作約 ${T} 公尺。`;
      const busPhrase = getNextBusArrivalPhrase(currStep.index);

      if (busPhrase && T <= 20) {
        // 距離<=20公尺時，立即啟動公車辨識（顯示手機後置鏡頭畫面）
        // 將臨界值從5公尺提高到20公尺，確保更容易觸發
        if (!busVisionActive) {
          if (T <= 5) {
            speak(`即將抵達公車站牌。${busPhrase}`);
          } else {
            speak(`距離下一個動作約 ${T} 公尺。${busPhrase}`);
          }
          // 立即啟動，不等待
          console.log(`[app] 距離<=${T}公尺，立即啟動公車辨識`);
          startBusVisionIfNeeded().catch(err => {
            console.error("[app] 啟動公車辨識失敗:", err);
          });
        }
      } else if (busPhrase) {
        speak(baseMsg + busPhrase);
      } else {
        speak(baseMsg);
      }
    }
  }
  
  // 額外檢查：如果距離已經<=20公尺但還沒有啟動，立即啟動（不依賴閾值觸發）
  // 將臨界值從5公尺提高到20公尺，確保更容易觸發
  const busPhrase = getNextBusArrivalPhrase(currStep.index);
  if (busPhrase && distance <= 20 && !busVisionActive) {
    console.log(`[app] 距離<=${Math.round(distance)}公尺但尚未啟動辨識，立即啟動`);
    startBusVisionIfNeeded().catch(err => {
      console.error("[app] 啟動公車辨識失敗:", err);
    });
  }

  lastDistanceToTarget = distance;
}

// ==== Haversine for GPS ====

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

// ==== 模擬路線 ====

// 解碼 Google Maps polyline
function decodePolyline(encoded) {
  if (!encoded || !window.google || !window.google.maps) {
    return null;
  }
  try {
    return window.google.maps.geometry.encoding.decodePath(encoded);
  } catch (e) {
    console.error("decodePolyline error:", e);
    return null;
  }
}

// 在 polyline 上找到最接近目標距離的點
function findPointOnPolyline(polyline, targetDistance) {
  if (!polyline || polyline.length < 2) return null;
  
  let accumulatedDistance = 0;
  for (let i = 0; i < polyline.length - 1; i++) {
    const p1 = polyline[i];
    const p2 = polyline[i + 1];
    const segmentDistance = haversineDistance(p1.lat(), p1.lng(), p2.lat(), p2.lng());
    
    if (accumulatedDistance + segmentDistance >= targetDistance) {
      // 在這個線段上插值
      const t = (targetDistance - accumulatedDistance) / segmentDistance;
      const lat = p1.lat() + (p2.lat() - p1.lat()) * t;
      const lng = p1.lng() + (p2.lng() - p1.lng()) * t;
      return { lat, lng };
    }
    accumulatedDistance += segmentDistance;
  }
  
  // 如果目標距離超過總長度，返回最後一個點
  const last = polyline[polyline.length - 1];
  return { lat: last.lat(), lng: last.lng() };
}

async function simulateWalkStep(step) {
  const d = step.distance_m || 200;
  const segments = 20;

  const start =
    currentSimLatLng ||
    currentOrigin || {
      lat: FIXED_ORIGIN.lat,
      lng: FIXED_ORIGIN.lng,
    };
  const end = {
    lat: step.target_lat ?? start.lat,
    lng: step.target_lng ?? start.lng,
  };

  spokenThresholdsPerStep[step.index] = {};
  lastDistanceToTarget = d + 20;

  // 嘗試使用路線的 polyline 來精準定位
  let polylinePath = null;
  if (currentRoute && currentRoute.overview_polyline && window.google && window.google.maps) {
    polylinePath = decodePolyline(currentRoute.overview_polyline);
  }

  // 如果沒有 polyline，使用簡單的線性插值
  if (!polylinePath || polylinePath.length < 2) {
    for (let i = 0; i <= segments; i++) {
      const t = i / segments;
      const lat = start.lat + (end.lat - start.lat) * t;
      const lng = start.lng + (end.lng - start.lng) * t;
      const remaining = d * (1 - t);

      await sendLocationUpdate(lat, lng, step.index);
      handleDistanceHints(remaining);

      const baseMs = 900; // 基準 0.9 秒
      await sleep(baseMs / simSpeedFactor);
    }
  } else {
    // 使用 polyline 來精準定位
    // 計算從起點到終點的總距離
    let totalDistance = 0;
    for (let i = 0; i < polylinePath.length - 1; i++) {
      totalDistance += haversineDistance(
        polylinePath[i].lat(),
        polylinePath[i].lng(),
        polylinePath[i + 1].lat(),
        polylinePath[i + 1].lng()
      );
    }
    
    // 在 polyline 上均勻分佈點
    for (let i = 0; i <= segments; i++) {
      const t = i / segments;
      const targetDistance = totalDistance * t;
      const point = findPointOnPolyline(polylinePath, targetDistance);
      
      if (point) {
        const remaining = d * (1 - t);
        await sendLocationUpdate(point.lat, point.lng, step.index);
        handleDistanceHints(remaining);
      } else {
        // 降級到簡單插值
        const lat = start.lat + (end.lat - start.lat) * t;
        const lng = start.lng + (end.lng - start.lng) * t;
        const remaining = d * (1 - t);
        await sendLocationUpdate(lat, lng, step.index);
        handleDistanceHints(remaining);
      }

      const baseMs = 900; // 基準 0.9 秒
      await sleep(baseMs / simSpeedFactor);
    }
  }

  currentSimLatLng = end;

    // 如果下一步是搭公車，就在這裡等待公車辨識成功再繼續模擬
  const nextBusStep = getNextBusStep(step.index);
  if (nextBusStep) {
    await waitForBusDetection();
  }
}

async function simulateBusStep(step) {
  const start =
    currentSimLatLng ||
    currentOrigin || {
      lat: FIXED_ORIGIN.lat,
      lng: FIXED_ORIGIN.lng,
    };
  const end = {
    lat: step.target_lat ?? start.lat,
    lng: step.target_lng ?? start.lng,
  };

  const segments = 30; // 增加分段數，讓移動更平滑

  // 嘗試使用路線的 polyline 來精準定位（沿著公車預計行走的軌跡）
  let polylinePath = null;
  if (currentRoute && currentRoute.overview_polyline && window.google && window.google.maps) {
    polylinePath = decodePolyline(currentRoute.overview_polyline);
  }

  // 如果沒有 polyline，使用簡單的線性插值
  if (!polylinePath || polylinePath.length < 2) {
    // 降級到簡單插值
    for (let i = 0; i <= segments; i++) {
      const t = i / segments;
      const lat = start.lat + (end.lat - start.lat) * t;
      const lng = start.lng + (end.lng - start.lng) * t;

      await sendLocationUpdate(lat, lng, step.index);

      const baseMs = 750;
      await sleep(baseMs / simSpeedFactor);
    }
  } else {
    // 使用 polyline 來精準定位，沿著公車預計行走的軌跡
    // 計算從起點到終點的總距離
    let totalDistance = 0;
    for (let i = 0; i < polylinePath.length - 1; i++) {
      totalDistance += haversineDistance(
        polylinePath[i].lat(),
        polylinePath[i].lng(),
        polylinePath[i + 1].lat(),
        polylinePath[i + 1].lng()
      );
    }
    
    // 在 polyline 上均勻分佈點，沿著公車路線移動
    for (let i = 0; i <= segments; i++) {
      const t = i / segments;
      const targetDistance = totalDistance * t;
      const point = findPointOnPolyline(polylinePath, targetDistance);
      
      if (point) {
        await sendLocationUpdate(point.lat, point.lng, step.index);
      } else {
        // 降級到簡單插值
        const lat = start.lat + (end.lat - start.lat) * t;
        const lng = start.lng + (end.lng - start.lng) * t;
        await sendLocationUpdate(lat, lng, step.index);
      }

      const baseMs = 750;
      await sleep(baseMs / simSpeedFactor);
    }
  }

  currentSimLatLng = end;
}

async function simulateCurrentRoute() {
  if (!hasRoute || !currentRoute || !currentRoute.steps) {
    speak("尚未取得路線，請先使用 Google 規劃路線。");
    return;
  }
  if (isSimulating) {
    speak("已經在模擬當中。");
    return;
  }

  if (simSpeedInput) {
    const v = parseFloat(simSpeedInput.value);
    if (!Number.isNaN(v) && v > 0.1) {
      simSpeedFactor = v;
    }
  }

  isSimulating = true;
  speak("開始模擬完整路線。");
  spokenThresholdsPerStep = {};
  lastDistanceToTarget = null;

  if (!currentOrigin) {
    currentOrigin = { ...FIXED_ORIGIN };
  }
  currentSimLatLng = { ...currentOrigin };

  try {
    for (const step of currentRoute.steps) {
      currentStepIndex = step.index;
      updateStepInfo();

      // 一開始先講「這一步要做什麼」
      if (step.instruction) {
        if (step.mode === "arrive") {
          // 到達時稍微完整一點
          speak(step.instruction);
        } else {
          speak(`第 ${step.index + 1} 步。${step.instruction}`);
        }
      }

      if (step.mode === "walk") {
        await simulateWalkStep(step);
      } else if (step.mode === "bus_ride") {
        await simulateBusStep(step);
      } else if (step.mode === "arrive") {
        await sleep(1500 / simSpeedFactor);
      } else {
        await sleep(1000 / simSpeedFactor);
      }
    }

    speak("整條路線模擬完成。");
  } catch (e) {
    console.error("simulateCurrentRoute error:", e);
    speak("模擬過程中發生錯誤。");
  } finally {
    isSimulating = false;
  }
}

// ==== 起點 & 出發時間 ====

function parseDepartureDateTimeInput() {
  if (!departureTimeInput || !departureTimeInput.value) return null;

  const timeStr = departureTimeInput.value;
  const timeParts = timeStr.split(":");
  if (timeParts.length !== 2) return null;
  const hh = parseInt(timeParts[0], 10);
  const mm = parseInt(timeParts[1], 10);
  if (Number.isNaN(hh) || Number.isNaN(mm)) return null;

  let baseDate;
  if (departureDateInput && departureDateInput.value) {
    const dParts = departureDateInput.value.split("-");
    if (dParts.length === 3) {
      const year = parseInt(dParts[0], 10);
      const monthIdx = parseInt(dParts[1], 10) - 1;
      const day = parseInt(dParts[2], 10);
      if (
        !Number.isNaN(year) &&
        !Number.isNaN(monthIdx) &&
        !Number.isNaN(day)
      ) {
        baseDate = new Date(year, monthIdx, day, hh, mm, 0, 0);
      }
    }
  }
  if (!baseDate) {
    const now = new Date();
    baseDate = new Date(
      now.getFullYear(),
      now.getMonth(),
      now.getDate(),
      hh,
      mm,
      0,
      0
    );
  }

  const now = new Date();
  if (baseDate.getTime() < now.getTime()) {
    return Math.floor(now.getTime() / 1000);
  }
  return Math.floor(baseDate.getTime() / 1000);
}

// 重置所有狀態和快取
async function resetAllState() {
  console.log("[reset] 開始重置所有狀態...");
  
  // 停止模擬
  isSimulating = false;
  
  // 停止公車辨識和手機視訊流
  if (busVisionActive) {
    busVisionActive = false;
    busVisionSpoken = false;
    waitingForBusDetection = false;
    stopMobileStream();
    fetch("/bus_vision/stop", { method: "POST" }).catch(console.error);
  }
  
  // 重置公車辨識狀態（後端）
  fetch("/bus_vision/reset", { method: "POST" }).catch(console.error);
  
  // 重置路線相關狀態
  currentRoute = null;
  hasRoute = false;
  currentStepIndex = 0;
  spokenThresholdsPerStep = {};
  lastDistanceToTarget = null;
  
  // 重置位置到清大原點
  currentOrigin = { lat: FIXED_ORIGIN.lat, lng: FIXED_ORIGIN.lng };
  currentSimLatLng = { lat: FIXED_ORIGIN.lat, lng: FIXED_ORIGIN.lng };
  
  // 更新顯示
  updateStepInfo();
  updateStatus("已重置，準備開始新的導航。");
  
  console.log("[reset] 重置完成");
}

async function doStartGoogleWithOrigin(originLat, originLng, destText, depEpoch) {
  // 先重置所有狀態
  await resetAllState();
  
  // 設定新的起點
  currentOrigin = { lat: originLat, lng: originLng };
  currentSimLatLng = { lat: originLat, lng: originLng };

  const route = await requestGoogleRoute(originLat, originLng, destText, depEpoch);
  if (!route) {
    speak("路線規劃失敗，請稍後再試。");
    return;
  }

  currentRoute = route;
  hasRoute = true;
  currentStepIndex = 0;
  spokenThresholdsPerStep = {};
  lastDistanceToTarget = null;

  updateStepInfo();
  
  // 構建詳細的語音提示
  let routeMessage = "";
  if (currentRoute.summary) {
    routeMessage = currentRoute.summary;
  } else {
    routeMessage = `已規劃前往 ${destText} 的路線。`;
  }
  
  // 添加預計時間資訊
  if (currentRoute.duration_text) {
    routeMessage += ` 預計行程時間 ${currentRoute.duration_text}。`;
  }
  
  // 添加距離資訊
  if (currentRoute.distance_text) {
    routeMessage += ` 總距離 ${currentRoute.distance_text}。`;
  }
  
  speak(routeMessage);
}

async function startGoogleNavigation() {
  const destText = destInput?.value?.trim();
  if (!destText) {
    speak("請先輸入目的地。");
    return;
  }

  // 添加語音提示：正在規劃路線
  speak(`正在規劃前往 ${destText} 的路線，請稍候。`);

  const depEpoch = parseDepartureDateTimeInput();

  if (USE_FIXED_ORIGIN) {
    await doStartGoogleWithOrigin(
      FIXED_ORIGIN.lat,
      FIXED_ORIGIN.lng,
      destText,
      depEpoch
    );
  } else {
    if (!navigator.geolocation) {
      speak("此裝置不支援定位功能。");
      return;
    }
    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        const lat = pos.coords.latitude;
        const lng = pos.coords.longitude;
        await doStartGoogleWithOrigin(lat, lng, destText, depEpoch);
      },
      (err) => {
        console.error("getCurrentPosition error:", err);
        speak("取得目前位置失敗。");
      },
      { enableHighAccuracy: true, timeout: 10000 }
    );
  }
}

// 等待公車辨識成功（有任何 bus_number 就算成功）
async function waitForBusDetection() {
  if (waitingForBusDetection) {
    console.log("[bus_detection] 已經在等待公車辨識中");
    return;
  }
  waitingForBusDetection = true;
  busVisionSpoken = false; // 重置狀態，確保可以重新偵測

  console.log("[bus_detection] 開始等待公車辨識");
  
  // 確保已經啟動公車辨識（啟動手機後置鏡頭）
  await startBusVisionIfNeeded();

  speak("已抵達公車站，正在辨識公車車號，請把鏡頭對準公車前方的電子看板。需要收集5個偵測結果才能繼續。");

  // 開始輪詢公車辨識狀態
  pollBusVisionStatusLoop();

  // 等待直到收集到5個偵測結果並確定最有信心的號碼（busVisionSpoken 變為 true）
  let waitCount = 0;
  const maxWait = 300; // 最多等待 5 分鐘（300 * 1秒）
  while (!busVisionSpoken && waitCount < maxWait) {
    await sleep(1000);
    waitCount++;
    
    // 每5秒檢查一次進度
    if (waitCount % 5 === 0) {
      try {
        const resp = await fetch("/bus_vision/status");
        if (resp.ok) {
          const s = await resp.json();
          const count = (s.all_detections || []).length;
          console.log(`[bus_detection] 等待中... 已收集 ${count}/5 個偵測結果 (${waitCount}/${maxWait}秒)`);
          if (count < 5) {
            speak(`已收集 ${count} 個偵測結果，還需要 ${5 - count} 個。`);
          }
        }
      } catch (e) {
        console.error("[bus_detection] 檢查進度失敗:", e);
      }
    }
  }

  if (busVisionSpoken) {
    console.log("[bus_detection] 公車辨識成功，繼續模擬");
  } else {
    console.log("[bus_detection] 等待超時，繼續模擬");
    speak("公車辨識超時，繼續導航。");
  }

  // 關閉手機相機和公車監視視訊
  stopMobileStream();
  fetch("/bus_vision/stop", { method: "POST" }).catch(console.error);
  busVisionActive = false;

  waitingForBusDetection = false;
}

async function startBusVisionIfNeeded() {
  if (busVisionActive) {
    console.log("[bus_vision] 公車辨識已經啟動");
    return;
  }
  
  console.log("[bus_vision] 準備啟動公車辨識（手機後置鏡頭）");
  
  try {
    // 檢查是否支援 getUserMedia（支援多種瀏覽器）
    let getUserMedia = null;
    
    // 優先使用標準 API
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      getUserMedia = (constraints) => navigator.mediaDevices.getUserMedia(constraints);
      console.log("[bus_vision] 使用標準 mediaDevices.getUserMedia");
    }
    // 降級到舊版 API（相容舊瀏覽器）
    else if (navigator.getUserMedia) {
      getUserMedia = (constraints) => {
        return new Promise((resolve, reject) => {
          navigator.getUserMedia(constraints, resolve, reject);
        });
      };
      console.log("[bus_vision] 使用舊版 navigator.getUserMedia");
    }
    // 降級到 webkit 前綴（Safari 舊版）
    else if (navigator.webkitGetUserMedia) {
      getUserMedia = (constraints) => {
        return new Promise((resolve, reject) => {
          navigator.webkitGetUserMedia(constraints, resolve, reject);
        });
      };
      console.log("[bus_vision] 使用 webkitGetUserMedia");
    }
    // 降級到 moz 前綴（Firefox 舊版）
    else if (navigator.mozGetUserMedia) {
      getUserMedia = (constraints) => {
        return new Promise((resolve, reject) => {
          navigator.mozGetUserMedia(constraints, resolve, reject);
        });
      };
      console.log("[bus_vision] 使用 mozGetUserMedia");
    }
    
    if (!getUserMedia) {
      // 檢測瀏覽器類型
      const userAgent = navigator.userAgent.toLowerCase();
      let browserName = "未知瀏覽器";
      let suggestion = "";
      
      if (userAgent.includes("chrome") && !userAgent.includes("edg")) {
        browserName = "Chrome";
        suggestion = "請確認您使用的是 Chrome 瀏覽器（不是其他基於 Chromium 的瀏覽器），並且版本是最新的。";
      } else if (userAgent.includes("safari") && !userAgent.includes("chrome")) {
        browserName = "Safari";
        suggestion = "請確認您使用的是 Safari 瀏覽器，並且 iOS 版本是最新的。";
      } else if (userAgent.includes("firefox")) {
        browserName = "Firefox";
        suggestion = "請確認您使用的是 Firefox 瀏覽器，並且版本是最新的。";
      } else if (userAgent.includes("samsung")) {
        browserName = "Samsung Internet";
        suggestion = "Samsung Internet 瀏覽器可能不支援，請改用 Chrome 瀏覽器。";
      } else if (userAgent.includes("ucbrowser") || userAgent.includes("uc browser")) {
        browserName = "UC Browser";
        suggestion = "UC Browser 不支援相機功能，請改用 Chrome 或 Firefox 瀏覽器。";
      } else if (userAgent.includes("qqbrowser") || userAgent.includes("mqqbrowser")) {
        browserName = "QQ Browser";
        suggestion = "QQ Browser 可能不支援，請改用 Chrome 或 Firefox 瀏覽器。";
      } else {
        suggestion = "請使用 Chrome、Firefox 或 Safari 瀏覽器，並確保使用最新版本。";
      }
      
      const errorMsg = `您的瀏覽器（${browserName}）不支援相機功能。${suggestion}`;
      console.error("[bus_vision]", errorMsg);
      console.error("[bus_vision] 詳細瀏覽器資訊:", {
        userAgent: navigator.userAgent,
        mediaDevices: !!navigator.mediaDevices,
        getUserMedia: !!navigator.getUserMedia,
        webkitGetUserMedia: !!navigator.webkitGetUserMedia,
        mozGetUserMedia: !!navigator.mozGetUserMedia,
        protocol: window.location.protocol,
        hostname: window.location.hostname
      });
      
      updateCameraStatus(errorMsg, "error");
      speak(errorMsg);
      return;
    }
    
    // 檢查是否為 HTTPS 或 localhost（iOS 要求 HTTPS）
    const isSecure = window.location.protocol === "https:" || 
                     window.location.hostname === "localhost" || 
                     window.location.hostname === "127.0.0.1";
    if (!isSecure) {
      const errorMsg = "iOS 要求使用 HTTPS 才能開啟相機。請使用 ngrok 或設定 HTTPS 伺服器。目前連線： " + window.location.protocol + "//" + window.location.hostname;
      console.error("[bus_vision]", errorMsg);
      updateCameraStatus(errorMsg, "error");
      speak("iOS 要求使用 HTTPS 連線才能開啟相機，請使用 HTTPS 網址。");
      return;
    }
    
    // 先啟動手機後置鏡頭（用於拍攝公車）
    if (!mobileStream) {
      let streamError = null;
      
      // 優先嘗試使用後置鏡頭（environment）- 用於拍攝公車
      try {
        console.log("[bus_vision] 嘗試開啟後置鏡頭...");
        mobileStream = await getUserMedia({
          video: {
            facingMode: "environment", // 使用後置鏡頭（拍攝公車）
            width: { ideal: 640 },
            height: { ideal: 480 }
          }
        });
        console.log("[bus_vision] 後置鏡頭已開啟");
      } catch (err1) {
        console.warn("[bus_vision] 後置鏡頭開啟失敗，嘗試前置鏡頭...", err1);
        streamError = err1;
        
        // 如果後置鏡頭失敗，嘗試前置鏡頭
        try {
          mobileStream = await getUserMedia({
            video: {
              facingMode: "user", // 使用前置鏡頭
              width: { ideal: 640 },
              height: { ideal: 480 }
            }
          });
          console.log("[bus_vision] 前置鏡頭已開啟（後置鏡頭不可用）");
        } catch (err2) {
          console.warn("[bus_vision] 前置鏡頭也失敗，嘗試不指定 facingMode...", err2);
          
          // 最後嘗試：不指定 facingMode，讓系統自動選擇
          try {
            mobileStream = await getUserMedia({
              video: {
                width: { ideal: 640 },
                height: { ideal: 480 }
              }
            });
            console.log("[bus_vision] 相機已開啟（自動選擇）");
          } catch (err3) {
            console.error("[bus_vision] 所有嘗試都失敗:", err3);
            streamError = err3;
          }
        }
      }
      
      // 如果所有嘗試都失敗
      if (!mobileStream) {
        let errorMsg = "無法開啟手機鏡頭。";
        
        if (streamError) {
          const errorName = streamError.name;
          if (errorName === "NotAllowedError" || errorName === "PermissionDeniedError") {
            errorMsg = "無法開啟手機鏡頭：請在瀏覽器設定中允許相機權限。";
          } else if (errorName === "NotFoundError" || errorName === "DevicesNotFoundError") {
            errorMsg = "無法開啟手機鏡頭：找不到可用的相機裝置。";
          } else if (errorName === "NotReadableError" || errorName === "TrackStartError") {
            errorMsg = "無法開啟手機鏡頭：相機可能被其他應用程式使用中。";
          } else if (errorName === "OverconstrainedError" || errorName === "ConstraintNotSatisfiedError") {
            errorMsg = "無法開啟手機鏡頭：不支援要求的相機設定。";
          } else {
            errorMsg = `無法開啟手機鏡頭：${streamError.message || "未知錯誤"}`;
          }
        }
        
        console.error("[bus_vision]", errorMsg, streamError);
        speak(errorMsg);
        return;
      }
    }
    
    // 開始發送視訊流到後端
    startMobileStream();
    
    // 啟動後端辨識（使用手機模式）
    const resp = await fetch("/bus_vision/start?use_mobile=true", { 
      method: "POST" 
    });
    if (resp.ok) {
      busVisionActive = true;
      busVisionSpoken = false;
      // 不再在手機端打開視窗，改為在電腦端 monitor 自動打開
      speak("已開啟公車辨識模式，請把手機鏡頭對準即將進站的公車前方。公車監視畫面將在電腦端自動開啟。");
      pollBusVisionStatusLoop();
    } else {
      const errorText = await resp.text();
      console.error("[bus_vision] 後端啟動失敗:", resp.status, errorText);
      speak("後端辨識服務啟動失敗。");
      stopMobileStream();
    }
  } catch (e) {
    console.error("startBusVisionIfNeeded error:", e);
    speak("開啟公車辨識模式時發生錯誤。");
    stopMobileStream();
  }
}

function startMobileStream() {
  if (!mobileStream || mobileStreamInterval) return;
  
  const video = document.createElement("video");
  video.srcObject = mobileStream;
  video.autoplay = true;
  video.playsInline = true; // iOS 需要這個屬性
  
  // 確保 video 元素播放
  video.play().then(() => {
    console.log("[bus_vision] Video element 開始播放");
    updateCameraStatus("相機已開啟並正在發送視訊流", "success");
  }).catch((err) => {
    console.error("[bus_vision] Video play 失敗:", err);
    updateCameraStatus("相機開啟但播放失敗: " + err.message, "error");
  });
  
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  // 適度提高解析度以改善流暢度（在網路負擔和流暢度之間平衡）
  canvas.width = 640;  // 從 480 提高到 640
  canvas.height = 480; // 從 360 提高到 480
  
  let frameCount = 0;
  let lastSendTime = 0;
  
  // 每 100ms 發送一次 frame（約 10 FPS，提高流暢度）
  mobileStreamInterval = setInterval(async () => {
    if (!busVisionActive || !mobileStream) {
      stopMobileStream();
      return;
    }
    
    // 節流：確保不會發送太快
    const now = Date.now();
    if (now - lastSendTime < 100) {
      return;
    }
    lastSendTime = now;
    
    try {
      // 檢查 video 是否已載入
      if (video.readyState < 2) {
        console.warn("[bus_vision] Video 尚未就緒，跳過此 frame");
        return;
      }
      
      // 將 video frame 繪製到 canvas
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      // 轉換為 blob（適度提高品質以改善流暢度）
      canvas.toBlob(async (blob) => {
        if (!blob) {
          console.warn("[bus_vision] Canvas toBlob 失敗");
          return;
        }
        
        const formData = new FormData();
        formData.append("file", blob, "frame.jpg");
        
        try {
          // 使用 AbortController 設定超時，避免卡住
          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), 3000); // 3秒超時
          
          const resp = await fetch("/bus_vision/mobile_frame", {
            method: "POST",
            body: formData,
            signal: controller.signal,
          });
          
          clearTimeout(timeoutId);
          
          if (resp.ok) {
            frameCount++;
            if (frameCount % 10 === 0) {
              console.log(`[bus_vision] 已發送 ${frameCount} 個 frames`);
            }
          } else {
            console.warn("[bus_vision] 後端接收 frame 失敗:", resp.status);
          }
        } catch (err) {
          if (err.name === 'AbortError') {
            console.warn("[bus_vision] 發送 frame 超時");
          } else {
            console.error("[bus_vision] 發送 frame 失敗:", err);
          }
        }
      }, "image/jpeg", 0.75); // 適度提高品質以改善流暢度（從 0.6 提高到 0.75）
    } catch (err) {
      console.error("[bus_vision] 處理 frame 失敗:", err);
    }
  }, 100); // 100ms 發送一次（約 10 FPS，提高流暢度）
  
  console.log("[bus_vision] 開始發送手機視訊流");
  updateCameraStatus("相機已開啟，正在發送視訊流...", "info");
}

function stopMobileStream() {
  if (mobileStreamInterval) {
    clearInterval(mobileStreamInterval);
    mobileStreamInterval = null;
  }
  
  if (mobileStream) {
    mobileStream.getTracks().forEach(track => {
      track.stop();
      console.log("[bus_vision] 已停止 track:", track.kind, track.label);
    });
    mobileStream = null;
  }
  
  console.log("[bus_vision] 已停止手機視訊流");
  updateCameraStatus("相機已關閉", "info");
}

function updateCameraStatus(message, type = "info") {
  const statusDiv = document.getElementById("camera-status");
  const statusText = document.getElementById("camera-status-text");
  
  if (statusDiv && statusText) {
    statusDiv.style.display = "block";
    statusText.textContent = message;
    
    // 根據類型設定顏色
    if (type === "success") {
      statusText.style.color = "#22c55e";
    } else if (type === "error") {
      statusText.style.color = "#ef4444";
    } else {
      statusText.style.color = "#a5b4fc";
    }
  }
}

// 測試相機功能
async function testCamera() {
  updateCameraStatus("正在測試相機...", "info");
  
  try {
    // 檢查是否支援 getUserMedia（支援多種瀏覽器）
    let getUserMedia = null;
    
    // 優先使用標準 API
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
      getUserMedia = (constraints) => navigator.mediaDevices.getUserMedia(constraints);
      console.log("[test] 使用標準 mediaDevices.getUserMedia");
    }
    // 降級到舊版 API（相容舊瀏覽器）
    else if (navigator.getUserMedia) {
      getUserMedia = (constraints) => {
        return new Promise((resolve, reject) => {
          navigator.getUserMedia(constraints, resolve, reject);
        });
      };
      console.log("[test] 使用舊版 navigator.getUserMedia");
    }
    // 降級到 webkit 前綴（Safari 舊版）
    else if (navigator.webkitGetUserMedia) {
      getUserMedia = (constraints) => {
        return new Promise((resolve, reject) => {
          navigator.webkitGetUserMedia(constraints, resolve, reject);
        });
      };
      console.log("[test] 使用 webkitGetUserMedia");
    }
    // 降級到 moz 前綴（Firefox 舊版）
    else if (navigator.mozGetUserMedia) {
      getUserMedia = (constraints) => {
        return new Promise((resolve, reject) => {
          navigator.mozGetUserMedia(constraints, resolve, reject);
        });
      };
      console.log("[test] 使用 mozGetUserMedia");
    }
    
    if (!getUserMedia) {
      // 檢測瀏覽器類型
      const userAgent = navigator.userAgent.toLowerCase();
      let browserName = "未知瀏覽器";
      let suggestion = "";
      
      if (userAgent.includes("chrome") && !userAgent.includes("edg")) {
        browserName = "Chrome";
        suggestion = "請確認您使用的是 Chrome 瀏覽器（不是其他基於 Chromium 的瀏覽器），並且版本是最新的。";
      } else if (userAgent.includes("safari") && !userAgent.includes("chrome")) {
        browserName = "Safari";
        suggestion = "請確認您使用的是 Safari 瀏覽器，並且 iOS 版本是最新的。";
      } else if (userAgent.includes("firefox")) {
        browserName = "Firefox";
        suggestion = "請確認您使用的是 Firefox 瀏覽器，並且版本是最新的。";
      } else if (userAgent.includes("samsung")) {
        browserName = "Samsung Internet";
        suggestion = "Samsung Internet 瀏覽器可能不支援，請改用 Chrome 瀏覽器。";
      } else if (userAgent.includes("ucbrowser") || userAgent.includes("uc browser")) {
        browserName = "UC Browser";
        suggestion = "UC Browser 不支援相機功能，請改用 Chrome 或 Firefox 瀏覽器。";
      } else if (userAgent.includes("qqbrowser") || userAgent.includes("mqqbrowser")) {
        browserName = "QQ Browser";
        suggestion = "QQ Browser 可能不支援，請改用 Chrome 或 Firefox 瀏覽器。";
      } else {
        suggestion = "請使用 Chrome、Firefox 或 Safari 瀏覽器，並確保使用最新版本。";
      }
      
      const errorMsg = `您的瀏覽器（${browserName}）不支援相機功能。${suggestion}`;
      updateCameraStatus(errorMsg, "error");
      console.error("[test]", errorMsg);
      console.error("[test] 詳細瀏覽器資訊:", {
        userAgent: navigator.userAgent,
        mediaDevices: !!navigator.mediaDevices,
        getUserMedia: !!navigator.getUserMedia,
        webkitGetUserMedia: !!navigator.webkitGetUserMedia,
        mozGetUserMedia: !!navigator.mozGetUserMedia,
        protocol: window.location.protocol,
        hostname: window.location.hostname
      });
      speak(errorMsg);
      return;
    }
    
    // 先停止現有的 stream（如果有的話）
    if (mobileStream) {
      stopMobileStream();
      await new Promise(resolve => setTimeout(resolve, 500));
    }
    
    // 嘗試開啟後置鏡頭
    console.log("[test] 嘗試開啟後置鏡頭...");
    updateCameraStatus("正在請求相機權限...", "info");
    
    const stream = await getUserMedia({
      video: {
        facingMode: "environment", // 後置鏡頭
        width: { ideal: 640 },
        height: { ideal: 480 }
      }
    });
    
    console.log("[test] 相機已開啟，tracks:", stream.getTracks().map(t => `${t.kind}:${t.label}`));
    updateCameraStatus("✓ 相機測試成功！後置鏡頭已開啟", "success");
    speak("相機測試成功，後置鏡頭已開啟");
    
    // 顯示相機預覽（可選）
    const video = document.createElement("video");
    video.srcObject = stream;
    video.autoplay = true;
    video.playsInline = true;
    video.style.width = "100%";
    video.style.maxWidth = "300px";
    video.style.borderRadius = "8px";
    video.style.marginTop = "8px";
    
    const statusDiv = document.getElementById("camera-status");
    if (statusDiv) {
      // 移除舊的 video（如果有的話）
      const oldVideo = statusDiv.querySelector("video");
      if (oldVideo) oldVideo.remove();
      statusDiv.appendChild(video);
    }
    
    video.play().then(() => {
      console.log("[test] 預覽播放成功");
    }).catch(err => {
      console.error("[test] 預覽播放失敗:", err);
    });
    
    // 5 秒後自動關閉測試
    setTimeout(() => {
      stream.getTracks().forEach(track => track.stop());
      const videoEl = statusDiv?.querySelector("video");
      if (videoEl) videoEl.remove();
      updateCameraStatus("測試完成，相機已關閉", "info");
    }, 5000);
    
  } catch (err) {
    console.error("[test] 相機測試失敗:", err);
    let errorMsg = "相機測試失敗：";
    
    if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
      errorMsg = "相機權限被拒絕，請在瀏覽器設定中允許相機權限";
    } else if (err.name === "NotFoundError") {
      errorMsg = "找不到相機裝置";
    } else if (err.name === "NotReadableError") {
      errorMsg = "相機可能被其他應用程式使用中";
    } else {
      errorMsg = err.message || "未知錯誤";
    }
    
    updateCameraStatus(errorMsg, "error");
    speak(errorMsg);
  }
}
async function pollBusVisionStatusLoop() {
  // 已經講過結果就不用一直念
  if (busVisionSpoken) return;

  try {
    const resp = await fetch("/bus_vision/status");
    if (resp.ok) {
      const s = await resp.json();
      
      // 檢查是否已經收集到5個偵測結果
      const detectionCount = s.detection_count || 0;
      const requiredCount = s.required_count || 5;
      const allDetections = s.all_detections || [];  // 所有5個原始偵測結果
      const has5Detections = detectionCount >= requiredCount;
      const hasEnough = s.has_enough_detections || has5Detections;
      
      // 當收集到5個時，強制從中選出最有信心的號碼（即使信心度很低也要選）
      let bestNumber = s.best_bus_number;
      if (hasEnough && allDetections.length >= 5 && !bestNumber) {
        // 如果還沒有選出最有信心的，強制從5個中選一個（選信心度最高的）
        const bestItem = allDetections.reduce((max, item) => 
          item.confidence > max.confidence ? item : max
        );
        bestNumber = bestItem.bus_number;
        console.log(`[bus_detection] 強制選出最有信心號碼: ${bestNumber}`);
      }
      
      console.log(`[bus_detection] 目前偵測結果數: ${detectionCount}/${requiredCount}, 最有信心號碼: ${bestNumber}`);
      
      // 當收集到5個偵測結果時，檢查信心度是否達到90%
      if (hasEnough && allDetections.length >= 5) {
        // 如果還沒有選出最有信心的，從5個中選信心度最高的
        if (!bestNumber && allDetections.length > 0) {
          const bestItem = allDetections.reduce((max, item) => 
            item.confidence > max.confidence ? item : max
          );
          bestNumber = bestItem.bus_number;
        }
        
        // 檢查信心度是否達到90%，並且驗證號碼是否有效（不能是"0"或空字串）
        const isValidNumber = bestNumber && bestNumber !== "0" && bestNumber.trim() !== "" && /^\d{1,4}$/.test(bestNumber);
        const bestConf = isValidNumber ? (allDetections.find(d => d.bus_number === bestNumber)?.confidence || s.best_confidence || 0) : 0;
        
        if (isValidNumber && bestConf >= 0.90) {
          // 信心度達到90%且號碼有效，立即停止AI模型辨識並清除快取
          if (busVisionActive) {
            stopMobileStream();
            
            // 停止AI模型並清除快取
            fetch("/bus_vision/stop", { method: "POST" }).catch(console.error);
            fetch("/bus_vision/reset", { method: "POST" }).catch(console.error);
            
            busVisionActive = false;
            console.log(`[bus_detection] 已收集到5個偵測結果，信心度達到90% (${(bestConf * 100).toFixed(1)}%)，號碼: ${bestNumber}，立即停止AI模型辨識並清除快取`);
          }
          
          // 等待一下確保AI模型完全停止
          await sleep(500);
          
          // 通知 bus_monitor 顯示成功訊息
          const busMonitorWin = window.open("", "bus_monitor");
          if (busMonitorWin && !busMonitorWin.closed) {
            busMonitorWin.postMessage({
              type: "detection_success",
              busNumber: bestNumber
            }, "*");
          }
          
          // 假設 currentStepIndex 對應的下一個 bus_ride step 就是你要搭的
          const busStep = getNextBusStep(currentStepIndex);
          const expected = busStep ? busStep.bus_number : null;

          let msg = "";
          if (expected) {
            if (bestNumber.toString() === expected.toString()) {
              msg = `已辨識到公車 ${bestNumber}，這是你要搭的班次。`;
            } else {
              msg = `已辨識到公車 ${bestNumber}，不是你要搭的班次 ${expected}。請不要上車。`;
            }
          } else {
            msg = `已辨識到公車 ${bestNumber}。`;
          }

          const tsNow = Math.floor(Date.now() / 1000);
          const busStep2 = getNextBusStep(currentStepIndex);
          if (busStep2 && busStep2.bus_departure_timestamp) {
            const remainMin = Math.max(
              0,
              Math.round((busStep2.bus_departure_timestamp - tsNow) / 60)
            );
            if (remainMin > 0) {
              msg += ` 預計大約 ${remainMin} 分鐘後發車。`;
            } else {
              msg += " 現在隨時可能發車，請準備上車。";
            }
          }

          speak(msg);
          busVisionSpoken = true;
          console.log(`[bus_detection] 辨識成功，AI模型已關閉，快取已清除，立即繼續模擬`);
          
          // 立即繼續模擬，不要等待
          // waitForBusDetection 會返回，然後 simulateWalkStep 會繼續執行下一個步驟
          return;
        } else {
          console.log("[bus_detection] 已收集到5個偵測結果，但無法確定最有信心的號碼，繼續等待...");
        }
      } else {
        // 還沒收集到5個，繼續等待
        const count = allDetections.length;
        if (count > 0 && count % 1 === 0) { // 每收集到一個就提示
          console.log(`[bus_detection] 已收集 ${count}/5 個偵測結果，繼續等待...`);
        }
      }
    }
  } catch (e) {
    console.error("pollBusVisionStatusLoop error:", e);
  }

  // 還沒收集到5個偵測結果，就過 1 秒再問一次（只在 busVisionActive 時持續）
  if (busVisionActive && !busVisionSpoken) {
    setTimeout(pollBusVisionStatusLoop, 1000); // 改為1秒檢查一次，更快響應
  }
}


// ==== 觸控板 ====

function setupTouchArea() {
  if (!touchArea) return;

  touchArea.addEventListener("click", (ev) => {
    ev.preventDefault();
    if (!hasRoute || !currentRoute || !currentRoute.steps) {
      speak("尚未開始導航。");
      return;
    }
    const s = currentRoute.steps[currentStepIndex];
    speak(s.instruction || "目前沒有可朗讀的指令。");
  });

  touchArea.addEventListener("dblclick", (ev) => {
    ev.preventDefault();
    if (!hasRoute || !currentRoute || !currentRoute.steps) return;
    if (currentStepIndex < currentRoute.steps.length - 1) {
      currentStepIndex += 1;
      spokenThresholdsPerStep[currentStepIndex] = {};
      lastDistanceToTarget = null;
      updateStepInfo();
      const s = currentRoute.steps[currentStepIndex];
      speak(`下一步。${s.instruction || ""}`);
    } else {
      speak("已經是最後一步。");
    }
  });
}

// ==== init ====

function init() {
  destInput = document.getElementById("destination-input");
  departureDateInput = document.getElementById("departure-date");
  departureTimeInput = document.getElementById("departure-time");
  simSpeedInput = document.getElementById("sim-speed");
  btnGoogleRoute = document.getElementById("btn-google-route");
  btnDemoRoute = document.getElementById("btn-demo-route");
  btnSimulateRoute = document.getElementById("btn-simulate-route");
  btnEmergency = document.getElementById("btn-emergency");
  btnEnd = document.getElementById("btn-end");
  touchArea = document.getElementById("touch-area");
  statusLine = document.getElementById("status-line");
  stepInfo = document.getElementById("step-info");

  if (simSpeedInput) {
    simSpeedFactor = parseFloat(simSpeedInput.value) || 1.0;
    simSpeedInput.addEventListener("input", () => {
      const v = parseFloat(simSpeedInput.value);
      if (!Number.isNaN(v) && v > 0.1) {
        simSpeedFactor = v;
      }
    });
  }

  if (btnDemoRoute) {
    btnDemoRoute.addEventListener("click", (ev) => {
      ev.preventDefault();
      requestDemoRoute();
    });
  }

  if (btnGoogleRoute) {
    btnGoogleRoute.addEventListener("click", (ev) => {
      ev.preventDefault();
      startGoogleNavigation();
    });
  }

  if (btnSimulateRoute) {
    btnSimulateRoute.addEventListener("click", (ev) => {
      ev.preventDefault();
      simulateCurrentRoute();
    });
  }

  if (btnEnd) {
    btnEnd.addEventListener("click", async (ev) => {
      ev.preventDefault();
      
      // 重置所有狀態和快取
      await resetAllState();
      
      speak("已結束導航，所有狀態已重置。");
      updateStatus("導航已結束，已重置到原點。");
    });
  }

  if (btnEmergency) {
    btnEmergency.addEventListener("mousedown", (ev) => {
      ev.preventDefault();
      isEmergency = true;
      speak("已開啟緊急模式。");
    });
    btnEmergency.addEventListener("mouseup", (ev) => {
      ev.preventDefault();
      isEmergency = false;
      speak("關閉緊急模式。");
    });
  }

  // 測試相機按鈕
  const btnTestCamera = document.getElementById("btn-test-camera");
  if (btnTestCamera) {
    btnTestCamera.addEventListener("click", (ev) => {
      ev.preventDefault();
      testCamera();
    });
  }

  setupTouchArea();
  
  // 確保 device_id 已生成
  ensureDeviceId();
  
  // 立即更新顯示（不等待註冊完成）
  updateDeviceIdDisplay();
  
  // 設定複製功能
  setupCopyDeviceIdButton();
  
  // 註冊裝置（異步，不阻塞顯示）
  registerDevice();
}

// 更新 device_id 顯示
function updateDeviceIdDisplay() {
  const deviceIdDisplay = document.getElementById("device-id-display");
  if (deviceIdDisplay) {
    if (deviceId) {
      deviceIdDisplay.textContent = deviceId;
    } else {
      deviceIdDisplay.textContent = "尚未生成";
    }
  }
}

// 設定複製按鈕功能
function setupCopyDeviceIdButton() {
  const copyDeviceIdBtn = document.getElementById("copy-device-id");
  if (copyDeviceIdBtn) {
    copyDeviceIdBtn.addEventListener("click", () => {
      // 確保 device_id 已生成
      if (!deviceId) {
        ensureDeviceId();
      }
      
      if (deviceId) {
        // 優先使用 Clipboard API
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(deviceId).then(() => {
            copyDeviceIdBtn.textContent = "已複製！";
            setTimeout(() => {
              copyDeviceIdBtn.textContent = "複製 ID";
            }, 2000);
          }).catch((err) => {
            console.error("Clipboard API 失敗:", err);
            // 降級到傳統方法
            fallbackCopyText(deviceId, copyDeviceIdBtn);
          });
        } else {
          // 降級到傳統方法
          fallbackCopyText(deviceId, copyDeviceIdBtn);
        }
      } else {
        alert("裝置 ID 尚未生成，請稍候再試");
      }
    });
  }
}

// 傳統複製方法（降級方案）
function fallbackCopyText(text, button) {
  try {
    const textArea = document.createElement("textarea");
    textArea.value = text;
    textArea.style.position = "fixed";
    textArea.style.left = "-999999px";
    textArea.style.top = "-999999px";
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    
    const successful = document.execCommand("copy");
    document.body.removeChild(textArea);
    
    if (successful) {
      button.textContent = "已複製！";
      setTimeout(() => {
        button.textContent = "複製 ID";
      }, 2000);
    } else {
      // 如果都失敗，顯示 device_id 讓用戶手動複製
      alert(`請手動複製以下裝置 ID：\n\n${text}`);
    }
  } catch (err) {
    console.error("複製失敗:", err);
    alert(`請手動複製以下裝置 ID：\n\n${text}`);
  }
}

document.addEventListener("DOMContentLoaded", init);

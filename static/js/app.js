// グローバル変数
let selectedFile = null;
let meterType = 'analog';

// DOM要素
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewArea = document.getElementById('previewArea');
const previewImage = document.getElementById('previewImage');
const btnClear = document.getElementById('btnClear');
const btnAnalyze = document.getElementById('btnAnalyze');
const loading = document.getElementById('loading');
const resultSection = document.getElementById('resultSection');
const errorSection = document.getElementById('errorSection');

// イベントリスナー設定
function setupEventListeners() {
    // メータータイプ選択
    document.querySelectorAll('input[name="meter_type"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            meterType = e.target.value;
        });
    });

    // アップロードエリアのイベント
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    uploadArea.addEventListener('click', () => fileInput.click());

    // ファイル選択
    fileInput.addEventListener('change', handleFileSelect);

    // クリアボタン
    btnClear.addEventListener('click', clearImage);

    // 解析ボタン
    btnAnalyze.addEventListener('click', analyzeMeter);
}

// ドラッグオーバー処理
function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.add('drag-over');
}

// ドラッグリーブ処理
function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('drag-over');
}

// ドロップ処理
function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    uploadArea.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        validateAndPreviewFile(file);
    }
}

// ファイル選択処理
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        validateAndPreviewFile(file);
    }
}

// ファイル検証とプレビュー
function validateAndPreviewFile(file) {
    // ファイルタイプ検証
    const allowedTypes = ['image/jpeg', 'image/png'];
    if (!allowedTypes.includes(file.type)) {
        showError({
            code: 'INVALID_FILE_TYPE',
            message: 'JPGまたはPNG形式の画像をアップロードしてください',
            details: `選択されたファイル形式: ${file.type}`
        });
        return;
    }

    // ファイルサイズ検証（10MB）
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
        showError({
            code: 'FILE_TOO_LARGE',
            message: '画像サイズは10MB以下にしてください',
            details: `選択されたファイルサイズ: ${(file.size / (1024 * 1024)).toFixed(2)}MB`
        });
        return;
    }

    // プレビュー表示
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        uploadArea.style.display = 'none';
        previewArea.style.display = 'block';
        btnAnalyze.disabled = false;
        hideError();
        hideResult();
    };
    reader.readAsDataURL(file);
}

// 画像クリア
function clearImage() {
    selectedFile = null;
    previewImage.src = '';
    fileInput.value = '';
    uploadArea.style.display = 'flex';
    previewArea.style.display = 'none';
    btnAnalyze.disabled = true;
    hideError();
    hideResult();
}

// メーター解析実行
async function analyzeMeter() {
    if (!selectedFile) return;

    // UI更新
    btnAnalyze.disabled = true;
    loading.style.display = 'flex';
    hideError();
    hideResult();

    // FormData作成
    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('meter_type', meterType);

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            displayResult(data.data);
        } else {
            showError(data.error);
        }
    } catch (error) {
        showError({
            code: 'NETWORK_ERROR',
            message: 'ネットワークエラーが発生しました',
            details: '接続を確認して再度お試しください'
        });
    } finally {
        loading.style.display = 'none';
        btnAnalyze.disabled = false;
    }
}

// 結果表示
function displayResult(data) {
    // メイン数値表示
    document.getElementById('resultValue').textContent = formatValue(data.value);
    document.getElementById('resultUnit').textContent = data.unit || '';

    // 信頼度表示
    const confidenceValue = document.getElementById('resultConfidence');
    const confidenceBadge = document.getElementById('confidenceBadge');
    confidenceValue.textContent = getConfidenceLabel(data.confidence);

    // 信頼度に応じたスタイル
    confidenceBadge.className = 'confidence-badge';
    if (data.confidence === 'high') {
        confidenceBadge.classList.add('confidence-high');
    } else if (data.confidence === 'medium') {
        confidenceBadge.classList.add('confidence-medium');
    } else {
        confidenceBadge.classList.add('confidence-low');
    }

    // 詳細情報
    const detailsContent = document.getElementById('resultDetailsContent');
    detailsContent.innerHTML = formatDetails(data.details, data.meter_type);

    // 結果セクションを表示
    resultSection.style.display = 'block';
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// 数値フォーマット
function formatValue(value) {
    if (value === null || value === undefined) {
        return '-';
    }
    if (typeof value === 'number') {
        return value.toLocaleString('ja-JP', { maximumFractionDigits: 3 });
    }
    return value.toString();
}

// 信頼度ラベル取得
function getConfidenceLabel(confidence) {
    const labels = {
        'high': '高',
        'medium': '中',
        'low': '低'
    };
    return labels[confidence] || confidence;
}

// 詳細情報フォーマット
function formatDetails(details, meterType) {
    if (!details) return '<p>詳細情報はありません</p>';

    let html = '<div class="details-grid">';

    if (meterType === 'analog') {
        if (details.scale_range) {
            html += `<div class="detail-item"><span class="detail-label">スケール範囲:</span> <span class="detail-value">${escapeHtml(details.scale_range)}</span></div>`;
        }
        if (details.needle_position) {
            html += `<div class="detail-item"><span class="detail-label">針の位置:</span> <span class="detail-value">${escapeHtml(details.needle_position)}</span></div>`;
        }
    } else if (meterType === 'digital_7segment') {
        if (details.decimal_places !== undefined) {
            html += `<div class="detail-item"><span class="detail-label">小数点以下桁数:</span> <span class="detail-value">${details.decimal_places}</span></div>`;
        }
        if (details.segment_status) {
            html += `<div class="detail-item"><span class="detail-label">セグメント状態:</span> <span class="detail-value">${escapeHtml(details.segment_status)}</span></div>`;
        }
    }

    if (details.notes) {
        html += `<div class="detail-item detail-item-full"><span class="detail-label">備考:</span> <span class="detail-value">${escapeHtml(details.notes)}</span></div>`;
    }

    html += '</div>';
    return html;
}

// HTMLエスケープ
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// エラー表示
function showError(error) {
    document.getElementById('errorMessage').textContent = error.message || 'エラーが発生しました';
    document.getElementById('errorDetails').textContent = error.details || '';
    errorSection.style.display = 'block';
    errorSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// エラー非表示
function hideError() {
    errorSection.style.display = 'none';
}

// 結果非表示
function hideResult() {
    resultSection.style.display = 'none';
}

// 初期化
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
});

// Prompt-Math editor front-end utilities with dynamic schedule configuration.

const PROMPT_MATH_API_URL = '/prompt-math/config';

const PROMPT_MATH_CONFIG_URL = (() => {
    if (typeof document !== 'undefined') {
        const script = document.currentScript;
        if (script && script.src) {
            try {
                return new URL('prompt_math_config.json', script.src).toString();
            } catch (error) {
                console.warn('PromptMath: unable to resolve config URL from script', error);
            }
        }
    }
    return 'prompt_math_config.json';
})();

const DEFAULT_SCHEDULE_FUNCTIONS = {
    fade_in: {
        direction: 'increase',
        defaults: { start: 0.0, end: 1.0 },
        parameters: [
            { name: 'start', type: 'float', required: true },
            { name: 'end', type: 'float', required: true },
            { name: 'curve', type: 'string', required: false },
        ],
        template: '@ fade_in({start}, {end})',
    },
    fade_out: {
        direction: 'decrease',
        defaults: { start: 0.0, end: 1.0 },
        parameters: [
            { name: 'start', type: 'float', required: true },
            { name: 'end', type: 'float', required: true },
            { name: 'curve', type: 'string', required: false },
        ],
        template: '@ fade_out({start}, {end})',
    },
};

const DEFAULT_TEMPLATES = [
    { name: 'Basic Analogy', code: '[[ [king] - [man] + [woman] ]]', description: 'Classic vector arithmetic analogy' },
    { name: 'Quality Aggregate', code: '[[ mean([blurry],[grainy],[ugly]) ]]', description: 'Average multiple negative qualities' },
    { name: 'Style Transfer', code: '[[ [content] + 0.7*([style] - mean([photo],[realistic])) ]]', description: 'Transfer style while preserving content' },
    { name: 'Temporal Fade In', code: '[[ [detailed] @ fade_in(0.2, 0.8) ]]', description: 'Gradually introduce details during sampling' },
    { name: 'Style Morphing', code: '[[ [oil_painting] @ fade_out(0.0, 0.5) + [watercolor] @ fade_in(0.5, 1.0) ]]', description: 'Transition from one style to another' },
    { name: 'Emphasis Burst', code: '[[ [sharp] @ emphasis(2.0, 0.0, 0.3) ]]', description: 'Strong emphasis early in sampling' },
    { name: 'Bell Curve Focus', code: '[[ [glowing] @ bell(0.5, 0.2, 2.0) ]]', description: 'Peak attention in the middle of sampling' },
    { name: 'Pulsing Effect', code: '[[ [dynamic] @ pulse(3, 0.5) ]]', description: 'Oscillating attention throughout sampling' },
];

const DEFAULT_PROMPT_MATH_CONFIG = {
    scheduleFunctions: DEFAULT_SCHEDULE_FUNCTIONS,
    samplers: {
        source: 'fallback',
        available: [],
        metadata: {},
    },
    templates: DEFAULT_TEMPLATES,
};

const CURVE_RESOLVERS = {
    linear: (t) => t,
    smooth: (t) => t * t * (3 - 2 * t),
    ease_in: (t) => t * t,
    ease_out: (t) => 1 - Math.pow(1 - t, 2),
    ease_in_out: (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2),
    constant: () => 0,
};

const clamp01 = (value) => Math.min(1, Math.max(0, value));

function sanitiseNumber(value, fallback) {
    if (typeof value === 'number' && Number.isFinite(value)) {
        return value;
    }
    if (typeof value === 'string') {
        const parsed = Number.parseFloat(value.trim());
        if (Number.isFinite(parsed)) {
            return parsed;
        }
    }
    return fallback;
}

function stripQuotes(value) {
    if (typeof value !== 'string') {
        return value;
    }
    const trimmed = value.trim();
    if ((trimmed.startsWith('"') && trimmed.endsWith('"')) || (trimmed.startsWith("'") && trimmed.endsWith("'"))) {
        return trimmed.slice(1, -1);
    }
    return trimmed;
}

function buildScheduleFunctionFactories(metadata = {}) {
    const factories = {};
    Object.entries(metadata).forEach(([name, info]) => {
        const defaults = info?.defaults ?? {};
        const parameters = info?.parameters ?? [];
        const direction = (info?.direction || 'increase').toLowerCase();
        const clampOutput = info?.clamp !== false;
        factories[name] = (...args) => {
            const resolved = {};
            parameters.forEach((param, index) => {
                const raw = args[index];
                let value;
                if (param.type === 'float') {
                    value = sanitiseNumber(raw, defaults[param.name]);
                    if ((value === undefined || value === null) && param.required) {
                        value = defaults[param.name] ?? 0;
                    }
                } else if (param.type === 'string') {
                    const candidate = raw !== undefined ? raw : defaults[param.name];
                    value = stripQuotes(candidate ?? '');
                } else {
                    value = raw ?? defaults[param.name];
                }
                resolved[param.name] = value;
            });

            const start = sanitiseNumber(resolved.start, defaults.start ?? 0.0) ?? 0.0;
            const end = sanitiseNumber(resolved.end, defaults.end ?? 1.0) ?? 1.0;
            const span = Math.max(end - start, 1e-6);
            const curveKey = stripQuotes(resolved.curve ?? defaults.curve ?? 'linear').toLowerCase();
            const curve = CURVE_RESOLVERS[curveKey] || CURVE_RESOLVERS.linear;

            return (time) => {
                let weight;
                if (time <= start) {
                    weight = 0;
                } else if (time >= end) {
                    weight = 1;
                } else {
                    const normalised = clamp01((time - start) / span);
                    weight = curve(normalised);
                }
                if (direction === 'decrease') {
                    weight = 1 - weight;
                }
                if (clampOutput) {
                    weight = clamp01(weight);
                }
                return weight;
            };
        };
    });
    return factories;
}

function formatScheduleTemplate(info) {
    if (!info || !info.template) {
        return null;
    }
    const defaults = info.defaults || {};
    return info.template.replace(/\{(\w+)\}/g, (_, key) => {
        const value = defaults[key];
        if (value === undefined || value === null) {
            if (key === 'curve') {
                return '"linear"';
            }
            return key === 'end' ? '1.0' : '0.0';
        }
        if (typeof value === 'number') {
            return value.toString();
        }
        return value;
    });
}

function mergePromptMathConfig(config) {
    const merged = JSON.parse(JSON.stringify(DEFAULT_PROMPT_MATH_CONFIG));
    if (config && typeof config === 'object') {
        if (config.scheduleFunctions && typeof config.scheduleFunctions === 'object') {
            merged.scheduleFunctions = {
                ...merged.scheduleFunctions,
                ...config.scheduleFunctions,
            };
        }
        if (config.samplers && typeof config.samplers === 'object') {
            const samplerConfig = config.samplers;
            const metadata = samplerConfig.metadata && typeof samplerConfig.metadata === 'object'
                ? samplerConfig.metadata
                : {};
            merged.samplers = {
                ...merged.samplers,
                ...samplerConfig,
                metadata: {
                    ...merged.samplers.metadata,
                    ...metadata,
                },
            };
            if (Array.isArray(samplerConfig.available)) {
                const ordered = [];
                const seen = new Set();
                samplerConfig.available.forEach((name) => {
                    if (typeof name === 'string' && !seen.has(name)) {
                        seen.add(name);
                        ordered.push(name);
                    }
                });
                merged.samplers.available = ordered;
            }
        }
        if (Array.isArray(config.templates)) {
            merged.templates = config.templates
                .filter((entry) => entry && typeof entry === 'object')
                .map((entry) => ({
                    name: entry.name || '',
                    code: entry.code || '',
                    description: entry.description || '',
                }));
        }
    }
    return merged;
}

async function fetchPromptMathConfigFromApi() {
    if (typeof fetch !== 'function') {
        return null;
    }
    try {
        const response = await fetch(PROMPT_MATH_API_URL, { cache: 'no-store' });
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.warn('PromptMath: API config unavailable', error);
        return null;
    }
}

async function fetchPromptMathConfigFromFile() {
    if (typeof fetch !== 'function') {
        return null;
    }
    try {
        const response = await fetch(PROMPT_MATH_CONFIG_URL, { cache: 'no-store' });
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.warn('PromptMath: static config unavailable', error);
        return null;
    }
}

async function loadPromptMathConfig() {
    if (typeof window !== 'undefined' && window.promptMathConfig) {
        return window.promptMathConfig;
    }
    let payload = await fetchPromptMathConfigFromApi();
    if (!payload) {
        payload = await fetchPromptMathConfigFromFile();
    }
    if (!payload) {
        console.warn('PromptMath: using built-in scheduler defaults');
    }
    const merged = mergePromptMathConfig(payload || {});
    if (typeof window !== 'undefined') {
        window.promptMathConfig = merged;
    }
    return merged;
}

class ExpressionCache {
    constructor(storageKey = 'prompt_math_cache') {
        this.storageKey = storageKey;
        this.maxEntries = 50;
        this._entries = this._load();
    }

    _load() {
        try {
            const raw = localStorage.getItem(this.storageKey);
            if (!raw) {
                return [];
            }
            const parsed = JSON.parse(raw);
            if (Array.isArray(parsed)) {
                return parsed;
            }
        } catch (error) {
            console.warn('PromptMath: failed to load editor cache', error);
        }
        return [];
    }

    _save() {
        try {
            localStorage.setItem(this.storageKey, JSON.stringify(this._entries));
        } catch (error) {
            console.warn('PromptMath: failed to persist cache', error);
        }
    }

    getAllExpressions() {
        return [...this._entries].sort((a, b) => (b.timestamp || 0) - (a.timestamp || 0));
    }

    addExpression(pseudo, expression) {
        const existingIndex = this._entries.findIndex((item) => item.pseudo === pseudo);
        const entry = {
            pseudo,
            expression,
            usageCount: 0,
            timestamp: Date.now(),
        };
        if (existingIndex >= 0) {
            this._entries.splice(existingIndex, 1, entry);
        } else {
            this._entries.unshift(entry);
            if (this._entries.length > this.maxEntries) {
                this._entries.pop();
            }
        }
        this._save();
    }

    clearCache() {
        this._entries = [];
        this._save();
    }

    recordUsage(pseudo) {
        const entry = this._entries.find((item) => item.pseudo === pseudo);
        if (entry) {
            entry.usageCount = (entry.usageCount || 0) + 1;
            entry.timestamp = Date.now();
            this._save();
        }
    }
}

class SchedulePreview {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
        this.timesteps = 30;
        this.schedules = [];
        if (this.canvas) {
            this.canvas.width = this.canvas.clientWidth || 320;
            this.canvas.height = this.canvas.clientHeight || 160;
        }
    }

    setTimesteps(steps) {
        this.timesteps = Math.max(Number.parseInt(steps, 10) || 1, 1);
        this.render();
    }

    clearSchedules() {
        this.schedules = [];
        this.render();
    }

    addSchedule(name, func, color) {
        this.schedules.push({ name, func, color });
        this.render();
    }

    render() {
        if (!this.ctx || !this.canvas) {
            return;
        }
        const ctx = this.ctx;
        const { width, height } = this.canvas;
        ctx.clearRect(0, 0, width, height);

        ctx.strokeStyle = '#3a3a3a';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(30, height - 20);
        ctx.lineTo(width - 10, height - 20);
        ctx.moveTo(30, height - 20);
        ctx.lineTo(30, 10);
        ctx.stroke();

        if (!this.schedules.length) {
            return;
        }

        const samples = Math.max(this.timesteps, 10);
        this.schedules.forEach(({ func, color }) => {
            ctx.strokeStyle = color || '#4a9eff';
            ctx.lineWidth = 2;
            ctx.beginPath();
            for (let i = 0; i <= samples; i += 1) {
                const t = i / samples;
                const weight = clamp01(func(t));
                const x = 30 + (width - 40) * t;
                const y = height - 20 - weight * (height - 40);
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();
        });
    }
}

class PromptMathEditor {
    constructor(config) {
        this.config = mergePromptMathConfig(config || {});
        this.scheduleMetadata = this.config.scheduleFunctions || {};
        this.scheduleFactories = buildScheduleFunctionFactories(this.scheduleMetadata);
        this.samplers = this.config.samplers || { available: [], metadata: {} };
        this.cache = new ExpressionCache();
        this.editor = null;
        this.schedulePreview = null;
        this.validationTimeout = null;
        this.currentTokenCounts = { clipL: 0, clipG: 0 };
        this.templates = this.initializeTemplates();

        this.init();
    }

    init() {
        this.initializeEditor();
        this.schedulePreview = new SchedulePreview('preview-canvas');
        this.setupEventListeners();
        this.initializeUI();
        this.loadCachedExpressions();
        console.log('Prompt-Math Editor initialised');
    }

    initializeEditor() {
        const textarea = document.getElementById('prompt-editor');
        if (!textarea) {
            console.warn('PromptMath: textarea with id "prompt-editor" not found.');
            return;
        }
        if (typeof CodeMirror === 'undefined') {
            console.warn('PromptMath: CodeMirror not available.');
            return;
        }
        this.editor = CodeMirror.fromTextArea(textarea, {
            mode: 'promptmath',
            theme: 'material-darker',
            lineNumbers: false,
            lineWrapping: true,
            matchBrackets: true,
            autoCloseBrackets: true,
            showHint: true,
            lint: true,
            gutters: ['CodeMirror-lint-markers'],
            extraKeys: {
                'Ctrl-Space': 'autocomplete',
                'Ctrl-/': 'toggleComment',
                'Ctrl-F': 'findPersistent',
                F11: (cm) => cm.setOption('fullScreen', !cm.getOption('fullScreen')),
                Esc: (cm) => {
                    if (cm.getOption('fullScreen')) {
                        cm.setOption('fullScreen', false);
                    }
                },
            },
        });

        this.editor.on('change', () => this.onEditorChange());
        this.editor.on('cursorActivity', () => this.updateCursorPosition());
        this.editor.on('focus', () => this.onEditorFocus());
        this.editor.on('blur', () => this.onEditorBlur());
    }

    setupEventListeners() {
        const bind = (id, handler) => {
            const element = document.getElementById(id);
            if (element) {
                element.addEventListener('click', handler);
            }
        };

        bind('parse-btn', () => this.parseExpression());
        bind('clear-btn', () => this.clearEditor());
        bind('cache-btn', () => this.showCacheModal());
        bind('help-btn', () => this.showHelpModal());
        bind('validate-btn', () => this.validateExpression());
        bind('template-btn', () => this.showTemplateModal());

        const saveConfirm = document.getElementById('save-confirm-btn');
        if (saveConfirm) {
            saveConfirm.addEventListener('click', () => this.saveExpression());
        }
        const saveCancel = document.getElementById('save-cancel-btn');
        if (saveCancel) {
            saveCancel.addEventListener('click', () => this.hideModal(document.getElementById('cache-modal')));
        }

        const clearCacheBtn = document.getElementById('clear-cache-btn');
        if (clearCacheBtn) {
            clearCacheBtn.addEventListener('click', () => this.clearCache());
        }

        const tabs = document.querySelectorAll('.tab-btn');
        tabs.forEach((btn) => {
            btn.addEventListener('click', (event) => this.switchTab(event.target.dataset.tab));
        });

        const functionItems = document.querySelectorAll('.function-item');
        functionItems.forEach((item) => {
            item.addEventListener('click', () => this.insertFunction(item.dataset.func));
        });

        const previewSteps = document.getElementById('preview-timesteps');
        if (previewSteps) {
            previewSteps.addEventListener('input', (event) => {
                const steps = Number.parseInt(event.target.value, 10) || 30;
                const label = document.getElementById('timesteps-value');
                if (label) {
                    label.textContent = steps;
                }
                if (this.schedulePreview) {
                    this.schedulePreview.setTimesteps(steps);
                }
            });
        }

        const encoderSelect = document.getElementById('encoder-select');
        if (encoderSelect) {
            encoderSelect.addEventListener('change', () => this.updateTokenCounts());
        }

        const poolingSelect = document.getElementById('pooling-select');
        if (poolingSelect) {
            poolingSelect.addEventListener('change', () => this.updateTokenCounts());
        }

        const modals = document.querySelectorAll('.modal .close');
        modals.forEach((closeBtn) => {
            closeBtn.addEventListener('click', (event) => {
                const modal = event.target.closest('.modal');
                this.hideModal(modal);
            });
        });

        document.querySelectorAll('.modal').forEach((modal) => {
            modal.addEventListener('click', (event) => {
                if (event.target === modal) {
                    this.hideModal(modal);
                }
            });
        });
    }

    initializeUI() {
        this.updateTokenCounts();
        this.updateCursorPosition();
        this.updateSyntaxStatus(true, 'Ready');
        this.renderSchedulerMetadata();
    }

    renderSchedulerMetadata() {
        const container = document.querySelector('[data-role="scheduler-list"]');
        if (!container) {
            return;
        }
        container.innerHTML = '';
        const samplerConfig = this.samplers || {};
        const available = Array.isArray(samplerConfig.available) ? samplerConfig.available : [];
        if (!available.length) {
            const empty = document.createElement('div');
            empty.className = 'sampler-empty';
            empty.textContent = 'No schedulers reported by ComfyUI.';
            container.appendChild(empty);
            return;
        }
        available.forEach((name) => {
            const meta = (samplerConfig.metadata && samplerConfig.metadata[name]) || {};
            const item = document.createElement('div');
            item.className = 'sampler-item';
            const title = document.createElement('div');
            title.className = 'sampler-name';
            title.textContent = meta.label || name;
            item.appendChild(title);
            if (meta.call) {
                const call = document.createElement('div');
                call.className = 'sampler-call';
                call.textContent = meta.call;
                item.appendChild(call);
            }
            if (Array.isArray(meta.parameters) && meta.parameters.length) {
                const params = document.createElement('div');
                params.className = 'sampler-parameters';
                params.textContent = 'Parameters: ' + meta.parameters.map((param) => param.name).join(', ');
                item.appendChild(params);
            }
            container.appendChild(item);
        });
    }

    onEditorChange() {
        this.updateTokenCounts();
        if (this.validationTimeout) {
            clearTimeout(this.validationTimeout);
        }
        this.validationTimeout = setTimeout(() => this.validateExpression(), 400);
        this.updateSchedulePreview();
    }

    onEditorFocus() {
        const wrapper = document.querySelector('.editor-wrapper');
        if (wrapper) {
            wrapper.classList.add('focused');
        }
    }

    onEditorBlur() {
        const wrapper = document.querySelector('.editor-wrapper');
        if (wrapper) {
            wrapper.classList.remove('focused');
        }
    }

    updateCursorPosition() {
        if (!this.editor) {
            return;
        }
        const position = this.editor.getCursor();
        const label = document.getElementById('cursor-position');
        if (label) {
            label.textContent = `Ln ${position.line + 1}, Col ${position.ch + 1}`;
        }
    }

    updateTokenCounts() {
        if (!this.editor) {
            return;
        }
        const text = this.editor.getValue();
        const tokens = (text.match(/\[[^\]]+\]/g) || []).length;
        const label = document.getElementById('token-count');
        if (label) {
            label.textContent = `${tokens} token${tokens === 1 ? '' : 's'}`;
        }
    }

    updateSyntaxStatus(isValid, message = 'Ready') {
        const status = document.getElementById('syntax-status');
        if (!status) {
            return;
        }
        if (isValid) {
            status.textContent = message;
            status.classList.remove('error');
            status.classList.add('ok');
        } else {
            status.textContent = message || 'Syntax error';
            status.classList.remove('ok');
            status.classList.add('error');
        }
    }

    validateExpression() {
        if (!this.editor) {
            return false;
        }
        const text = this.editor.getValue();
        const stack = [];
        const pairs = { '[': ']', '{': '}', '(': ')' };
        const openings = Object.keys(pairs);
        const closings = Object.values(pairs);

        for (const char of text) {
            if (openings.includes(char)) {
                stack.push(char);
            } else if (closings.includes(char)) {
                const expected = pairs[stack.pop()];
                if (expected !== char) {
                    this.updateSyntaxStatus(false, 'Mismatched brackets');
                    return false;
                }
            }
        }

        if (stack.length) {
            this.updateSyntaxStatus(false, 'Unclosed bracket detected');
            return false;
        }

        this.updateSyntaxStatus(true, 'Syntax OK');
        return true;
    }

    updateSchedulePreview() {
        if (!this.editor || !this.schedulePreview) {
            return;
        }
        const text = this.editor.getValue();
        this.schedulePreview.clearSchedules();

        const regex = /@\s*(\w+)\s*\([^)]*\)/g;
        const matches = [...text.matchAll(regex)];
        const colors = ['#4a9eff', '#6b73ff', '#96ceb4', '#feca57', '#ff6b6b'];

        matches.forEach((match, index) => {
            const funcName = match[1];
            const factory = this.scheduleFactories[funcName];
            if (!factory) {
                return;
            }
            const params = this.parseScheduleParams(match[0]);
            try {
                const scheduleFunc = factory(...params);
                this.schedulePreview.addSchedule(funcName, scheduleFunc, colors[index % colors.length]);
            } catch (error) {
                console.warn(`PromptMath: unable to preview schedule ${funcName}`, error);
            }
        });
    }

    parseScheduleParams(scheduleText) {
        const paramMatch = scheduleText.match(/\(([^)]*)\)/);
        if (!paramMatch) {
            return [];
        }
        const paramStr = paramMatch[1];
        if (!paramStr.trim()) {
            return [];
        }
        return paramStr.split(',').map((part) => {
            const trimmed = part.trim();
            const number = Number.parseFloat(trimmed);
            return Number.isFinite(number) ? number : trimmed;
        });
    }

    switchTab(tabName) {
        document.querySelectorAll('.tab-btn').forEach((btn) => btn.classList.remove('active'));
        const tabButton = document.querySelector(`[data-tab="${tabName}"]`);
        if (tabButton) {
            tabButton.classList.add('active');
        }
        document.querySelectorAll('.tab-panel').forEach((panel) => panel.classList.remove('active'));
        const panel = document.getElementById(`${tabName}-tab`);
        if (panel) {
            panel.classList.add('active');
        }
    }

    insertFunction(funcName) {
        if (!this.editor) {
            return;
        }
        const cursor = this.editor.getCursor();
        const functionTemplates = {
            mean: 'mean([], [])',
            unit: 'unit([])',
            lerp: 'lerp([], [], 0.5)',
            slerp: 'slerp([], [], 0.5)',
            emphasis: '@ emphasis(2.0)',
            bell: '@ bell(0.5, 0.2)',
            pulse: '@ pulse(2.0, 1.0)',
        };

        let insertText;
        const scheduleTemplate = formatScheduleTemplate(this.scheduleMetadata[funcName]);
        if (scheduleTemplate) {
            insertText = scheduleTemplate;
        } else {
            insertText = functionTemplates[funcName] || `${funcName}()`;
        }

        this.editor.replaceRange(insertText, cursor);

        const newCursor = this.editor.getCursor();
        if (insertText.includes('(')) {
            const offset = insertText.indexOf('(') + 1;
            this.editor.setCursor(newCursor.line, newCursor.ch - insertText.length + offset);
        }
        this.editor.focus();
    }

    parseExpression() {
        if (!this.editor) {
            return;
        }
        const text = this.editor.getValue().trim();
        if (!text) {
            this.showMessage('Please enter an expression to parse.', 'warning');
            return;
        }
        if (!this.validateExpression()) {
            this.showMessage('Please fix syntax errors before parsing.', 'error');
            return;
        }
        this.showMessage('Expression parsed successfully!', 'success');
        this.updateSchedulePreview();
    }

    clearEditor() {
        if (!this.editor) {
            return;
        }
        if (confirm('Are you sure you want to clear the editor?')) {
            this.editor.setValue('');
            this.editor.focus();
        }
    }

    showMessage(message, type = 'info') {
        const container = document.createElement('div');
        container.className = `message message-${type}`;
        container.textContent = message;
        container.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 20px;
            border-radius: 4px;
            color: white;
            font-weight: 500;
            z-index: 1001;
            animation: slideIn 0.3s ease;
        `;
        const colors = {
            success: '#4caf50',
            warning: '#ff9800',
            error: '#f44336',
            info: '#2196f3',
        };
        container.style.backgroundColor = colors[type] || colors.info;
        document.body.appendChild(container);
        setTimeout(() => {
            container.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => container.remove(), 280);
        }, 3000);
    }

    showModal(modalId) {
        const modal = typeof modalId === 'string' ? document.getElementById(modalId) : modalId;
        if (modal) {
            modal.style.display = 'block';
            document.body.style.overflow = 'hidden';
        }
    }

    hideModal(modal) {
        const target = typeof modal === 'string' ? document.getElementById(modal) : modal;
        if (target) {
            target.style.display = 'none';
            document.body.style.overflow = 'auto';
        }
    }

    showCacheModal() {
        this.loadCachedExpressions();
        this.showModal('cache-modal');
    }

    showHelpModal() {
        this.showModal('help-modal');
    }

    showTemplateModal() {
        this.loadTemplates();
        this.showModal('template-modal');
    }

    showSaveExpressionModal() {
        if (!this.editor) {
            return;
        }
        const text = this.editor.getValue().trim();
        if (!text) {
            this.showMessage('Please enter an expression to save.', 'warning');
            return;
        }
        const preview = document.getElementById('expr-preview');
        const nameInput = document.getElementById('pseudo-name');
        if (preview) {
            preview.value = text;
        }
        if (nameInput) {
            nameInput.value = '';
        }
        this.showModal('cache-modal');
    }

    saveExpression() {
        const pseudoInput = document.getElementById('pseudo-name');
        const expressionInput = document.getElementById('expr-preview');
        if (!pseudoInput || !expressionInput) {
            return;
        }
        const pseudo = pseudoInput.value.trim();
        const expression = expressionInput.value.trim();
        if (!pseudo) {
            this.showMessage('Please enter a pseudotoken name.', 'warning');
            return;
        }
        if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(pseudo)) {
            this.showMessage('Pseudotoken name must be a valid identifier.', 'error');
            return;
        }
        if (!expression) {
            this.showMessage('No expression to save.', 'warning');
            return;
        }
        this.cache.addExpression(pseudo, expression);
        this.loadCachedExpressions();
        this.hideModal('cache-modal');
        this.showMessage(`Expression saved as _${pseudo}_`, 'success');
    }

    loadCachedExpressions() {
        const container = document.getElementById('cache-list');
        if (!container) {
            return;
        }
        const entries = this.cache.getAllExpressions();
        container.innerHTML = '';
        if (!entries.length) {
            container.innerHTML = '<p style="color: #888; text-align: center; padding: 20px;">No cached expressions</p>';
            return;
        }
        entries.forEach((item) => {
            const node = document.createElement('div');
            node.className = 'cache-item';
            node.innerHTML = `
                <div class="cache-pseudo">_${item.pseudo}_</div>
                <div class="cache-expr">${item.expression}</div>
                <div class="cache-meta">Used ${item.usageCount || 0} times Ã¢â‚¬Â¢ ${new Date(item.timestamp || Date.now()).toLocaleDateString()}</div>
            `;
            node.addEventListener('click', () => this.insertCachedExpression(item.pseudo));
            container.appendChild(node);
        });
    }

    insertCachedExpression(pseudo) {
        if (!this.editor) {
            return;
        }
        this.cache.recordUsage(pseudo);
        const cursor = this.editor.getCursor();
        this.editor.replaceRange(`_${pseudo}_`, cursor);
        this.editor.focus();
        this.hideModal('cache-modal');
    }

    clearCache() {
        if (confirm('Are you sure you want to clear all cached expressions?')) {
            this.cache.clearCache();
            this.loadCachedExpressions();
            this.showMessage('Cache cleared successfully.', 'success');
        }
    }

    initializeTemplates() {
        const templates = Array.isArray(this.config.templates) && this.config.templates.length
            ? this.config.templates
            : DEFAULT_PROMPT_MATH_CONFIG.templates;
        return templates.map((template) => ({
            name: template.name || '',
            code: template.code || '',
            description: template.description || '',
        }));
    }

    loadTemplates() {
        const list = document.querySelector('.template-list');
        if (!list) {
            return;
        }
        list.innerHTML = '';
        this.templates.forEach((template) => {
            const node = document.createElement('div');
            node.className = 'template-item';
            node.innerHTML = `
                <div class="template-name">${template.name}</div>
                <div class="template-code">${template.code}</div>
                <div class="template-desc">${template.description}</div>
            `;
            node.addEventListener('click', () => this.insertTemplate(template.code));
            list.appendChild(node);
        });
    }

    insertTemplate(code) {
        if (!this.editor) {
            return;
        }
        const cursor = this.editor.getCursor();
        this.editor.replaceRange(code, cursor);
        this.editor.focus();
        this.hideModal('template-modal');
        this.showMessage('Template inserted successfully.', 'success');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const bootstrap = async () => {
        const config = await loadPromptMathConfig();
        window.promptMathConfig = config;
        window.promptMathEditor = new PromptMathEditor(config);
    };

    bootstrap().catch((error) => {
        console.error('PromptMath: bootstrap failed', error);
        const fallback = mergePromptMathConfig({});
        window.promptMathConfig = fallback;
        window.promptMathEditor = new PromptMathEditor(fallback);
    });
});

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        PromptMathEditor,
        loadPromptMathConfig,
        DEFAULT_PROMPT_MATH_CONFIG,
    };
}

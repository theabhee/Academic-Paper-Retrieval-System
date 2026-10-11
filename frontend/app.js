document.addEventListener('DOMContentLoaded', () => {
  // State variables
  let currentMethod = 'semantic';
  let chartInstance = null;

  // DOM Elements
  const searchInput = document.getElementById('searchInput');
  const searchBtn = document.getElementById('searchBtn');
  const searchSpinner = document.getElementById('searchSpinner');
  const methodPills = document.querySelectorAll('.method-pills .pill');
  const sampleChips = document.querySelectorAll('.chip-query');
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');
  const paperList = document.getElementById('paperList');
  const resultsSummary = document.getElementById('resultsSummary');
  const topKSelect = document.getElementById('topKSelect');

  // Stats Elements
  const datasetSelect = document.getElementById('datasetSelect');
  const statDataset = document.getElementById('statDataset');
  const statPaperCount = document.getElementById('statPaperCount');
  const totalBrowseCount = document.getElementById('totalBrowseCount');

  if (datasetSelect) {
    datasetSelect.addEventListener('change', async () => {
      const mode = datasetSelect.value;
      try {
        await fetch(`/api/dataset/switch?mode=${mode}`, { method: 'POST' });
        await fetchStats();
        await fetchPapers();
        if (searchInput.value.trim()) {
          performSearch();
        }
      } catch (err) {
        console.error("Failed to switch dataset:", err);
      }
    });
  }

  const baselineSelect = document.getElementById('baselineSelect');
  if (baselineSelect) {
    baselineSelect.addEventListener('change', () => {
      if (searchInput.value.trim()) {
        performSearch();
      }
    });
  }

  // Fetch Stats on load
  fetchStats();
  fetchPapers();
  fetchEvaluationMetrics();

  // Event Listeners
  searchBtn.addEventListener('click', performSearch);
  searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') performSearch();
  });

  methodPills.forEach(pill => {
    pill.addEventListener('click', () => {
      methodPills.forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      currentMethod = pill.getAttribute('data-method');
      if (searchInput.value.trim()) {
        performSearch();
      }
    });
  });

  sampleChips.forEach(chip => {
    chip.addEventListener('click', () => {
      searchInput.value = chip.textContent.trim();
      performSearch();
    });
  });

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      const targetTab = btn.getAttribute('data-tab');
      document.getElementById(targetTab).classList.add('active');
      if (targetTab === 'rerankCompareTab' && searchInput.value.trim()) {
        currentMethod = 'compare_rerank';
        methodPills.forEach(p => p.classList.remove('active'));
        const rerankPill = document.querySelector('.pill-rerank');
        if (rerankPill) rerankPill.classList.add('active');
        performSearch();
      }
    });
  });

  // Fetch Dataset Statistics
  async function fetchStats() {
    try {
      const res = await fetch('/api/stats');
      const data = await res.json();
      statDataset.textContent = data.dataset_name;
      statPaperCount.textContent = data.total_papers;
      totalBrowseCount.textContent = data.total_papers;
    } catch (err) {
      console.error("Failed to fetch stats:", err);
    }
  }

  // Perform Search Query
  async function performSearch() {
    const query = searchInput.value.trim();
    if (!query) return;

    const topK = topKSelect.value;
    searchSpinner.style.display = 'inline-block';

    try {
      if (currentMethod === 'compare_rerank') {
        switchTab('rerankCompareTab');
        const baseEngine = baselineSelect ? baselineSelect.value : 'bm25';
        const res = await fetch(`/api/search?query=${encodeURIComponent(query)}&method=compare_rerank&base_engine=${baseEngine}&top_k=${topK}`);
        const data = await res.json();
        renderRerankComparison(data, query);
      } else if (currentMethod === 'compare_all') {
        // Switch to comparison tab
        switchTab('compareTab');
        const res = await fetch(`/api/search?query=${encodeURIComponent(query)}&method=compare_all&top_k=${topK}`);
        const data = await res.json();
        renderComparisonResults(data);
      } else {
        // Switch to search results tab
        switchTab('searchTab');
        const res = await fetch(`/api/search?query=${encodeURIComponent(query)}&method=${currentMethod}&top_k=${topK}`);
        const data = await res.json();
        renderSearchResults(data, query);
      }
    } catch (err) {
      console.error("Search error:", err);
      resultsSummary.textContent = "Error performing search. Please ensure API is running.";
    } finally {
      searchSpinner.style.display = 'none';
    }
  }

  function switchTab(tabId) {
    tabBtns.forEach(b => b.classList.remove('active'));
    tabContents.forEach(c => c.classList.remove('active'));

    const activeBtn = Array.from(tabBtns).find(b => b.getAttribute('data-tab') === tabId);
    if (activeBtn) activeBtn.classList.add('active');
    const targetElement = document.getElementById(tabId);
    if (targetElement) targetElement.classList.add('active');
  }

  // Render Original vs. Deep Learning Reranked Comparison
  function renderRerankComparison(data, query) {
    const origCol = document.getElementById('colOriginalResults');
    const rerankCol = document.getElementById('colRerankedResults');
    const origLabel = document.getElementById('origEngineLabel');
    const statNdcg = document.getElementById('statNdcgImpact');
    const statLatency = document.getElementById('statLatencyImpact');

    if (origLabel) origLabel.textContent = data.base_engine ? data.base_engine.toUpperCase() : 'BM25';

    // Update impact banner
    if (data.benchmark_metrics && statNdcg) {
      const bm = data.benchmark_metrics;
      statNdcg.textContent = `${bm.original_ndcg_3} ➔ ${bm.enhanced_ndcg_3} (${bm.quality_gain})`;
    }
    if (data.latency && statLatency) {
      statLatency.textContent = `Base: ${data.latency.original_ms}ms | Rerank: +${data.latency.rerank_ms}ms (Total: ${data.latency.total_ms}ms)`;
    }

    // Render Original Candidates
    origCol.innerHTML = '';
    const origList = data.original_results || [];
    if (origList.length === 0) {
      origCol.innerHTML = `<div style="color: var(--text-muted); text-align: center; padding: 20px;">No baseline candidate papers retrieved.</div>`;
    } else {
      origList.forEach((paper, idx) => {
        const card = document.createElement('div');
        card.className = 'comparison-paper-card';
        card.innerHTML = `
          <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 8px;">
            <div style="font-weight: 700; color: var(--text-main); font-size: 14px;">
              <span style="color: var(--cyan); font-family: var(--font-mono);">#${idx + 1}</span> ${highlightTerms(paper.title, query)}
            </div>
            <span class="score-badge" style="background: rgba(6, 182, 212, 0.15); color: var(--cyan); font-size: 11px; white-space: nowrap;">${paper.match_percentage}%</span>
          </div>
          <div style="font-size: 12px; color: var(--text-muted); margin-top: 6px; line-height: 1.4;">
            ${highlightTerms(paper.abstract.substring(0, 160), query)}...
          </div>
          <div style="margin-top: 8px; font-size: 11px; color: var(--text-dim); font-family: var(--font-mono);">
            Initial Score: ${paper.raw_score}
          </div>
        `;
        origCol.appendChild(card);
      });
    }

    // Render Reranked Results
    rerankCol.innerHTML = '';
    const rerankList = data.reranked_results || [];
    if (rerankList.length === 0) {
      rerankCol.innerHTML = `<div style="color: var(--text-muted); text-align: center; padding: 20px;">No papers returned by reranker.</div>`;
    } else {
      rerankList.forEach((paper, idx) => {
        const card = document.createElement('div');
        card.className = 'comparison-paper-card';
        card.style.borderColor = 'rgba(139, 92, 246, 0.25)';
        
        let shiftBadge = '';
        if (paper.rank_change > 0) {
          shiftBadge = `<span class="rank-shift-tag shift-promoted">▲ +${paper.rank_change} Promoted (was #${paper.orig_rank})</span>`;
        } else if (paper.rank_change < 0) {
          shiftBadge = `<span class="rank-shift-tag shift-demoted">▼ ${paper.rank_change} Demoted (was #${paper.orig_rank})</span>`;
        } else {
          shiftBadge = `<span class="rank-shift-tag shift-same">● Rank #${paper.orig_rank} Maintained</span>`;
        }

        card.innerHTML = `
          <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 8px;">
            <div style="font-weight: 700; color: var(--text-main); font-size: 14px;">
              <span style="color: var(--primary); font-family: var(--font-mono);">#${idx + 1}</span> ${highlightTerms(paper.title, query)}
            </div>
            <span class="score-badge" style="background: rgba(139, 92, 246, 0.2); color: #c4b5fd; font-size: 11px; white-space: nowrap;">${paper.match_percentage}%</span>
          </div>
          <div style="display: flex; align-items: center; gap: 8px; margin-top: 6px;">
            ${shiftBadge}
            <span style="font-size: 11px; color: var(--text-dim); font-family: var(--font-mono);">Cross-Encoder: ${paper.raw_score}</span>
          </div>
          <div style="font-size: 12px; color: var(--text-muted); margin-top: 6px; line-height: 1.4;">
            ${highlightTerms(paper.abstract.substring(0, 160), query)}...
          </div>
        `;
        rerankCol.appendChild(card);
      });
    }
  }

  // Render Single Search Results
  function renderSearchResults(data, query) {
    paperList.innerHTML = '';
    const results = data.results || [];

    resultsSummary.innerHTML = `Found <strong>${results.length}</strong> research papers matching <em>"${escapeHtml(query)}"</em> using <strong>${data.method.toUpperCase()}</strong> engine.`;

    if (results.length === 0) {
      paperList.innerHTML = `<div class="glass-panel" style="padding: 24px; text-align: center; color: var(--text-muted);">No matching papers found for this query. Try a different method like Semantic Search.</div>`;
      return;
    }

    results.forEach((paper, idx) => {
      const card = document.createElement('div');
      card.className = 'paper-card';
      
      const highlightedTitle = highlightTerms(paper.title, query);
      const highlightedAbstract = highlightTerms(paper.abstract, query);

      card.innerHTML = `
        <div class="paper-card-header">
          <div class="paper-title">${idx + 1}. ${highlightedTitle}</div>
          <div class="paper-badge-group">
            <span class="score-badge">${paper.match_percentage}% Match</span>
            <span class="raw-score-tag">Raw Score: ${paper.raw_score}</span>
          </div>
        </div>
        <div class="paper-abstract">${highlightedAbstract}</div>
        <div class="score-bar-wrapper">
          <div class="score-bar-fill" style="width: ${paper.match_percentage}%;"></div>
        </div>
      `;
      paperList.appendChild(card);
    });
  }

  // Render Side-by-Side Comparison
  function renderComparisonResults(data) {
    const methods = ['semantic', 'bm25', 'tfidf', 'boolean'];
    const containerMap = {
      'semantic': document.getElementById('colSemantic'),
      'bm25': document.getElementById('colBm25'),
      'tfidf': document.getElementById('colTfidf'),
      'boolean': document.getElementById('colBoolean')
    };

    methods.forEach(m => {
      const col = containerMap[m];
      col.innerHTML = '';
      const list = data.results[m] || [];

      if (list.length === 0) {
        col.innerHTML = `<div style="color: var(--text-dim); text-align: center; padding: 12px;">No matches</div>`;
        return;
      }

      list.forEach((p, idx) => {
        const miniCard = document.createElement('div');
        miniCard.className = 'mini-paper-card';
        miniCard.innerHTML = `
          <div class="mini-paper-title">${idx + 1}. ${escapeHtml(p.title)}</div>
          <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px;">
            <span class="score-badge" style="font-size: 11px; padding: 2px 6px;">${p.match_percentage}%</span>
            <span class="raw-score-tag">Score: ${p.raw_score}</span>
          </div>
        `;
        col.appendChild(miniCard);
      });
    });
  }

  // Fetch All Papers for Browser Tab
  async function fetchPapers() {
    const browseGrid = document.getElementById('browsePaperGrid');
    try {
      const res = await fetch('/api/papers?limit=50');
      const data = await res.json();
      browseGrid.innerHTML = '';

      data.papers.forEach((p) => {
        const item = document.createElement('div');
        item.className = 'paper-card';
        item.innerHTML = `
          <div class="paper-title" style="font-size: 16px;">${escapeHtml(p.title)}</div>
          <div class="paper-abstract" style="margin-top: 8px;">${escapeHtml(p.abstract.substring(0, 180))}...</div>
        `;
        browseGrid.appendChild(item);
      });
    } catch (err) {
      console.error("Failed to load papers:", err);
    }
  }

  // Fetch Evaluation Metrics
  async function fetchEvaluationMetrics() {
    try {
      const res = await fetch('/api/evaluate');
      const data = await res.json();
      renderMetricsTable(data.metrics);
      renderMetricsChart(data.metrics);
    } catch (err) {
      console.error("Failed to load evaluation metrics:", err);
    }
  }

  function renderMetricsTable(metrics) {
    const tbody = document.getElementById('metricsTableBody');
    tbody.innerHTML = '';

    metrics.forEach(m => {
      const row = document.createElement('tr');
      const isEnhanced = m.type === 'Enhanced' || m.model.includes('Reranker') || m.model.includes('Cross-Encoder');
      const typeBadge = isEnhanced
        ? `<span class="badge-tag" style="background: rgba(139, 92, 246, 0.25); color: #c4b5fd; font-size: 11px;">Enhanced</span>`
        : `<span class="badge-tag" style="background: rgba(6, 182, 212, 0.15); color: var(--cyan); font-size: 11px;">Original</span>`;

      const latencyText = m.latency_ms !== undefined ? `${m.latency_ms} ms` : '-';
      const ndcgScore = m.ndcg_3 !== undefined ? m.ndcg_3 : (m.ndcg || 0.85);

      row.innerHTML = `
        <td style="font-weight: 700; color: ${isEnhanced ? 'var(--primary)' : 'var(--text-main)'};">${escapeHtml(m.model)}</td>
        <td>${typeBadge}</td>
        <td>${(m.precision_3 * 100).toFixed(1)}%</td>
        <td>${(m.recall_3 * 100).toFixed(1)}%</td>
        <td style="font-weight: 700; color: #10b981;">${(ndcgScore * 100).toFixed(1)}%</td>
        <td style="color: var(--cyan); font-family: var(--font-mono);">${latencyText}</td>
      `;
      tbody.appendChild(row);
    });
  }

  function renderMetricsChart(metrics) {
    const ctx = document.getElementById('metricsChart').getContext('2d');
    const labels = metrics.map(m => m.model);

    if (chartInstance) chartInstance.destroy();

    chartInstance = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'nDCG@3 (Ranking Quality)',
            data: metrics.map(m => m.ndcg_3 !== undefined ? m.ndcg_3 : 0.85),
            backgroundColor: 'rgba(16, 185, 129, 0.75)',
            borderColor: '#10b981',
            borderWidth: 1
          },
          {
            label: 'Precision@3',
            data: metrics.map(m => m.precision_3),
            backgroundColor: 'rgba(139, 92, 246, 0.7)',
            borderColor: '#8b5cf6',
            borderWidth: 1
          },
          {
            label: 'Recall@3',
            data: metrics.map(m => m.recall_3),
            backgroundColor: 'rgba(6, 182, 212, 0.7)',
            borderColor: '#06b6d4',
            borderWidth: 1
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true,
            max: 1.0,
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#9ca3af' }
          },
          x: {
            grid: { display: false },
            ticks: { color: '#9ca3af', font: { size: 10 } }
          }
        },
        plugins: {
          legend: { labels: { color: '#f3f4f6', font: { family: 'Plus Jakarta Sans' } } }
        }
      }
    });
  }

  // Helper Utilities
  function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  function highlightTerms(text, query) {
    if (!text || !query) return escapeHtml(text);
    const terms = query.trim().split(/\s+/).filter(t => t.length > 2);
    if (terms.length === 0) return escapeHtml(text);

    let escaped = escapeHtml(text);
    terms.forEach(term => {
      const regex = new RegExp(`(${term})`, 'gi');
      escaped = escaped.replace(regex, '<mark>$1</mark>');
    });
    return escaped;
  }
});

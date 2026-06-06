import type { Campaign, KOL, PipelineKOL, Todo, KpiSnapshot, AiSummary, PreviewReviewResult, RiskFlag } from './types';

export const campaigns: Campaign[] = [
  {
    id: 'c1',
    name: '2024 Summer Skincare Launch',
    brand: 'GlowLab',
    budget: 500000,
    spent: 320000,
    status: 'ongoing',
    startDate: '2024-06-01',
    endDate: '2024-08-31',
    manager: 'Alice Chen',
    targetKOLs: 15,
    engagedKOLs: 12,
    description: '夏日护肤新品上市推广，重点覆盖 18-30 岁女性用户。',
    objectives: ['品牌曝光 500W+', 'CVR > 3%', 'UGC 内容 200+']
  },
  {
    id: 'c2',
    name: 'Tech Review Series Q3',
    brand: 'NovaTech',
    budget: 800000,
    spent: 150000,
    status: 'draft',
    startDate: '2024-07-15',
    endDate: '2024-09-30',
    manager: 'Bob Liu',
    targetKOLs: 8,
    engagedKOLs: 2,
    description: '季度科技测评系列，聚焦数码 3C。',
    objectives: ['科技圈层渗透', '电商导流']
  },
  {
    id: 'c3',
    name: 'Festival Makeup Challenge',
    brand: 'ColorPop',
    budget: 300000,
    spent: 280000,
    status: 'review',
    startDate: '2024-05-01',
    endDate: '2024-06-20',
    manager: 'Alice Chen',
    targetKOLs: 20,
    engagedKOLs: 18,
    description: '节日彩妆挑战赛，KOL 发起仿妆挑战。',
    objectives: ['话题曝光 1000W+', '挑战参与 500+']
  }
];

export const kols: KOL[] = [
  {
    id: 'k1',
    name: '小美爱护肤',
    handle: '@xiaomei_skincare',
    platform: '小红书',
    followers: 450000,
    engagementRate: 0.068,
    category: '护肤/美妆',
    avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=k1',
    email: 'xiaomei@kol.com',
    priceRange: '¥30k-50k',
    location: '上海',
    notes: [
      { id: 'n1', author: 'Alice', content: '回复速度很快，合作意愿强', createdAt: '2024-06-10T10:00:00Z', type: 'comment' }
    ]
  },
  {
    id: 'k2',
    name: 'TechBro_阿伟',
    handle: '@techbro_awei',
    platform: 'Bilibili',
    followers: 1200000,
    engagementRate: 0.045,
    category: '科技/数码',
    avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=k2',
    email: 'awei@techbro.com',
    priceRange: '¥80k-120k',
    location: '深圳',
    notes: [
      { id: 'n2', author: 'Bob', content: '报价偏高，需议价', createdAt: '2024-06-12T14:00:00Z', type: 'comment' }
    ]
  },
  {
    id: 'k3',
    name: 'LisaFitness',
    handle: '@lisafit',
    platform: 'Instagram',
    followers: 890000,
    engagementRate: 0.032,
    category: '健身/生活方式',
    avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=k3',
    email: 'lisa@fit.com',
    priceRange: '$5k-8k',
    location: 'Los Angeles',
    notes: []
  }
];

const rf = (msg: string, level: 'critical' | 'warning' | 'info' = 'warning', category: RiskFlag['category'] = 'schedule'): RiskFlag => ({
  id: Math.random().toString(36).slice(2),
  level,
  message: msg,
  category
});

export const pipelines: Record<string, PipelineKOL[]> = {
  c1: [
    { kol: kols[0], status: 'content_creation', price: 40000, deliverables: ['1 篇图文', '1 条视频'], deadline: '2024-06-25', lastActivity: '2024-06-18', riskFlags: [rf('初稿延迟 2 天', 'warning', 'schedule')] },
    { kol: kols[1], status: 'negotiating', price: 100000, deliverables: ['1 条深度测评视频'], deadline: '2024-07-10', lastActivity: '2024-06-15', riskFlags: [rf('报价超出预算 25%', 'critical', 'communication')] },
    { kol: kols[2], status: 'preview_review', price: 60000, deliverables: ['3 条 Stories', '1 条 Reel'], deadline: '2024-06-20', lastActivity: '2024-06-19', riskFlags: [rf('Preview 缺少 CTA', 'warning', 'content')] }
  ],
  c2: [
    { kol: kols[1], status: 'outreach', price: 100000, deliverables: ['1 条深度测评视频'], deadline: '2024-08-15', lastActivity: '2024-06-20', riskFlags: [] }
  ],
  c3: [
    { kol: kols[0], status: 'published', price: 35000, deliverables: ['1 篇图文'], deadline: '2024-06-10', lastActivity: '2024-06-12', riskFlags: [] }
  ]
};

export const todos: Todo[] = [
  { id: 't1', title: '审核 LisaFitness 的 Preview', campaignId: 'c1', campaignName: '2024 Summer Skincare Launch', dueDate: '2024-06-21', priority: 'high', done: false },
  { id: 't2', title: '与 TechBro_阿伟 确认合同', campaignId: 'c1', campaignName: '2024 Summer Skincare Launch', dueDate: '2024-06-22', priority: 'high', done: false },
  { id: 't3', title: '提交 Festival Makeup Challenge 结案报告', campaignId: 'c3', campaignName: 'Festival Makeup Challenge', dueDate: '2024-06-23', priority: 'medium', done: false }
];

export const kpiSnapshot: KpiSnapshot = {
  activeCampaigns: 2,
  liveKOLs: 12,
  pendingReviews: 5,
  overdueTasks: 2
};

export const aiSummary: AiSummary = {
  headline: '本周有 2 个高风险项需要关注',
  points: [
    'TechBro_阿伟 报价超预算 25%，建议重新议价或更换备选 KOL',
    'LisaFitness 的 Preview 缺少 CTA 和产品标签，需退回修改',
    'Festival Makeup Challenge 即将到期，建议提前催稿未提交 KOL'
  ],
  suggestedAction: '立即处理预算超支风险，避免影响整体 ROI。'
};

export const previewReviewMock: PreviewReviewResult = {
  overall_result: 'conditional_pass',
  risk_level: 'warning',
  missing_requirements: ['未展示产品全成分表', '未提及「经皮肤科测试」'],
  matched_selling_points: ['SPF50+ 防晒', '轻薄质地', '适合敏感肌'],
  missing_cta: true,
  hashtag_check: {
    required: ['#GlowLab', '#夏日护肤', '#防晒必备'],
    missing: ['#GlowLab'],
    extra: ['#我的护肤日记']
  },
  banned_expression_check: {
    found: ['最好用的防晒霜', '100% 无副作用'],
    severity: 'warning'
  },
  tone_check: {
    expected: '专业、温和、可信',
    actual: '偏口语化，部分夸张',
    score: 72
  },
  factual_risk: ['「SPF50+ 持续 12 小时」未提供测试报告支持'],
  suggested_revisions: [
    { original: '最好用的防晒霜', suggestion: '个人非常推荐的防晒霜之一', reason: '避免绝对化用语，符合广告法' },
    { original: '100% 无副作用', suggestion: '成分温和，经测试不易致敏', reason: '「100%」属于极限词' }
  ],
  message_to_kol: '感谢提交！内容整体不错，请补充 #GlowLab 标签，替换两处极限词，并加上购买链接 CTA。修改后可直接发布。',
  reviewer_summary: '内容质量中等，粉丝互动预期良好，但需修改合规问题后方可发布。'
};
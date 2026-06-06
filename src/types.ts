export type CampaignStatus = 'draft' | 'ongoing' | 'review' | 'completed' | 'cancelled';
export type KOLStatus = 'outreach' | 'negotiating' | 'contracted' | 'content_creation' | 'preview_review' | 'published' | 'paused' | 'dropped';
export type RiskLevel = 'critical' | 'warning' | 'info';

export interface Campaign {
  id: string;
  name: string;
  brand: string;
  budget: number;
  spent: number;
  status: CampaignStatus;
  startDate: string;
  endDate: string;
  manager: string;
  targetKOLs: number;
  engagedKOLs: number;
  description: string;
  objectives: string[];
}

export interface KOL {
  id: string;
  name: string;
  handle: string;
  platform: 'Instagram' | 'TikTok' | 'YouTube' | '小红书' | '微博';
  followers: number;
  engagementRate: number;
  category: string;
  avatar: string;
  email: string;
  priceRange: string;
  location: string;
  notes: Note[];
}

export interface Note {
  id: string;
  author: string;
  content: string;
  createdAt: string;
  type: 'comment' | 'status_change' | 'ai_suggestion';
}

export interface PipelineKOL {
  kol: KOL;
  status: KOLStatus;
  price: number;
  deliverables: string[];
  deadline: string;
  lastActivity: string;
  riskFlags: RiskFlag[];
}

export interface RiskFlag {
  id: string;
  level: RiskLevel;
  message: string;
  category: 'schedule' | 'content' | 'compliance' | 'communication';
}

export interface Todo {
  id: string;
  title: string;
  campaignId: string;
  campaignName: string;
  dueDate: string;
  priority: 'high' | 'medium' | 'low';
  done: boolean;
}

export interface KpiSnapshot {
  activeCampaigns: number;
  liveKOLs: number;
  pendingReviews: number;
  overdueTasks: number;
}

export interface AiSummary {
  headline: string;
  points: string[];
  suggestedAction: string;
}

export interface PreviewReviewResult {
  overall_result: 'pass' | 'conditional_pass' | 'fail';
  risk_level: RiskLevel;
  missing_requirements: string[];
  matched_selling_points: string[];
  missing_cta: boolean;
  hashtag_check: { required: string[]; missing: string[]; extra: string[] };
  banned_expression_check: { found: string[]; severity: RiskLevel };
  tone_check: { expected: string; actual: string; score: number };
  factual_risk: string[];
  suggested_revisions: Array<{ original: string; suggestion: string; reason: string }>;
  message_to_kol: string;
  reviewer_summary: string;
}
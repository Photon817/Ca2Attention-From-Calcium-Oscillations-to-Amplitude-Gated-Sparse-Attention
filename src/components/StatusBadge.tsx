import type { CampaignStatus, KOLStatus, RiskLevel } from '../types';

const campaignMap: Record<CampaignStatus, string> = {
  draft: 'bg-slate-100 text-slate-700',
  ongoing: 'bg-brand-50 text-brand-700',
  review: 'bg-amber-50 text-amber-700',
  completed: 'bg-green-50 text-green-700',
  cancelled: 'bg-red-50 text-red-700',
};

const kolMap: Record<KOLStatus, string> = {
  outreach: 'bg-slate-100 text-slate-700',
  negotiating: 'bg-blue-50 text-blue-700',
  contracted: 'bg-brand-50 text-brand-700',
  content_creation: 'bg-purple-50 text-purple-700',
  preview_review: 'bg-amber-50 text-amber-700',
  published: 'bg-green-50 text-green-700',
  paused: 'bg-orange-50 text-orange-700',
  dropped: 'bg-red-50 text-red-700',
};

const riskMap: Record<RiskLevel, string> = {
  critical: 'bg-red-50 text-red-700 border-red-200',
  warning: 'bg-amber-50 text-amber-700 border-amber-200',
  info: 'bg-blue-50 text-blue-700 border-blue-200',
};

const labels: Record<string, string> = {
  draft: '草稿', ongoing: '进行中', review: '审核中', completed: '已完成', cancelled: '已取消',
  outreach: '触达中', negotiating: '议价中', contracted: '已签约', content_creation: '内容制作',
  preview_review: 'Preview 审核', published: '已发布', paused: '暂停', dropped: '已放弃',
  critical: '严重', warning: '警告', info: '提示'
};

export function CampaignBadge({ status }: { status: CampaignStatus }) {
  return <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${campaignMap[status]}`}>{labels[status]}</span>;
}

export function KOLBadge({ status }: { status: KOLStatus }) {
  return <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${kolMap[status]}`}>{labels[status]}</span>;
}

export function RiskBadge({ level }: { level: RiskLevel }) {
  return <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium border ${riskMap[level]}`}>{labels[level]}</span>;
}
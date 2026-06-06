import { useParams, Link } from 'react-router-dom';
import { CheckCircle2, XCircle, AlertCircle, ArrowLeft, Send } from 'lucide-react';
import Nav from '../components/Nav';
import { previewReviewMock, campaigns } from '../mockData';
import { RiskBadge } from '../components/StatusBadge';

export default function PreviewReview() {
  const { id } = useParams<{ id: string }>();
  const campaign = campaigns.find((c) => c.id === id);

  const r = previewReviewMock;

  return (
    <div>
      <Nav items={[{ label: 'Campaigns', to: '/campaigns' }, { label: campaign?.name || 'Campaign', to: `/campaigns/${id}` }, { label: 'Preview 审核' }]} />

      <div className="flex items-center justify-between mb-5">
        <h2 className="text-xl font-bold text-slate-900">Preview 审核</h2>
        <div className="flex items-center gap-2">
          <button className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg border border-slate-200 text-sm text-slate-600 hover:bg-slate-50">
            <ArrowLeft size={14} /> 退回修改
          </button>
          <button className="inline-flex items-center gap-1.5 px-3 py-2 rounded-lg bg-brand-600 text-white text-sm font-medium hover:bg-brand-700">
            <Send size={14} /> 发送给 KOL
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-white rounded-xl border border-slate-200 p-5">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                {r.overall_result === 'pass' ? <CheckCircle2 className="text-green-600" size={24} />
                  : r.overall_result === 'fail' ? <XCircle className="text-red-600" size={24} />
                  : <AlertCircle className="text-amber-600" size={24} />}
                <div>
                  <div className="text-sm font-semibold text-slate-900">Overall Result</div>
                  <div className="text-xs text-slate-500">{r.reviewer_summary}</div>
                </div>
              </div>
              <RiskBadge level={r.risk_level} />
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="rounded-lg bg-slate-50 p-3">
                <div className="text-xs text-slate-500 mb-1">Matched Selling Points</div>
                <div className="flex flex-wrap gap-1">
                  {r.matched_selling_points.map((s) => (
                    <span key={s} className="px-2 py-0.5 rounded bg-green-50 text-green-700 text-xs">{s}</span>
                  ))}
                </div>
              </div>
              <div className="rounded-lg bg-slate-50 p-3">
                <div className="text-xs text-slate-500 mb-1">Missing Requirements</div>
                <ul className="list-disc pl-4 text-xs text-red-700 space-y-0.5">
                  {r.missing_requirements.map((m) => <li key={m}>{m}</li>)}
                </ul>
              </div>
              <div className="rounded-lg bg-slate-50 p-3">
                <div className="text-xs text-slate-500 mb-1">Hashtag Check</div>
                <div className="text-xs space-y-1">
                  <div>缺失: {r.hashtag_check.missing.join(', ') || '无'}</div>
                  <div>多余: {r.hashtag_check.extra.join(', ') || '无'}</div>
                </div>
              </div>
              <div className="rounded-lg bg-slate-50 p-3">
                <div className="text-xs text-slate-500 mb-1">Banned Expressions</div>
                <div className="text-xs text-red-700 space-y-0.5">
                  {r.banned_expression_check.found.map((b) => <div key={b}>· {b}</div>)}
                </div>
              </div>
            </div>

            <div className="mt-4">
              <div className="text-xs text-slate-500 mb-1">Tone Check</div>
              <div className="text-sm text-slate-700">Score: <span className="font-semibold">{r.tone_check.score}</span>/100 · 期望: {r.tone_check.expected} · 实际: {r.tone_check.actual}</div>
            </div>

            <div className="mt-4">
              <div className="text-xs text-slate-500 mb-1">Suggested Revisions</div>
              <div className="space-y-2">
                {r.suggested_revisions.map((rev, i) => (
                  <div key={i} className="rounded-lg border border-slate-100 p-3 text-sm">
                    <div className="text-slate-400 line-through">{rev.original}</div>
                    <div className="text-green-700 font-medium">{rev.suggestion}</div>
                    <div className="text-xs text-slate-500 mt-0.5">{rev.reason}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="bg-white rounded-xl border border-slate-200 p-5">
            <h3 className="text-sm font-semibold text-slate-900 mb-2">发给 KOL 的话</h3>
            <p className="text-sm text-slate-700 leading-relaxed">{r.message_to_kol}</p>
            <button className="mt-3 w-full inline-flex items-center justify-center gap-1.5 px-3 py-2 rounded-lg bg-brand-600 text-white text-sm font-medium hover:bg-brand-700">
              <Send size={14} /> 一键发送
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
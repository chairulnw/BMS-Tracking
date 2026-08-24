import { Component, OnInit, inject } from '@angular/core';
import { NgClass } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { ActivatedRoute, Router } from '@angular/router';
import { forkJoin } from 'rxjs';
import { environment } from '../../../environments/environment';
import { AuthService } from '../../services/auth.service';
import { AuthUrlPipe } from '../../pipes/auth-url.pipe';

const API    = environment.apiBaseUrl;
const AI_API = environment.aiServiceBaseUrl;

// Ukuran bucket waktu di tab Timeline — satu konstanta, mudah diubah.
const BUCKET_MINUTES = 5;

interface PersonDetail {
  id: number;
  name:               string;
  label:              string;
  jabatan:            string | null;
  is_known:           boolean;
  last_camera:        string | null;
  last_camera_name:   string | null;
  last_seen:          string | null;
  best_thumbnail_url: string | null;
  observation_count:  number;
}

interface FeedItem {
  detection_id:    number;
  person_id:       number | null;
  camera_id:       string;
  camera_name:     string | null;
  camera_location: string | null;
  timestamp:       string;
  thumbnail_url:   string | null;
  tracklet_id:     number | null;
}

interface FeedResponse {
  items: FeedItem[];
  total: number;
  page:  number;
  pages: number;
}

interface CrossingItem {
  id:           number;
  timestamp:    string;
  direction:    string;
  camera_id:    string | null;
  camera_name:  string | null;
  zone_name:    string | null;
  snapshot_url: string | null;
}

type TimelineEntry =
  | { kind: 'detection'; key: string; timestamp: string; data: FeedItem }
  | { kind: 'event';     key: string; timestamp: string; data: CrossingItem };

interface BucketGroup {
  label: string;
  items: TimelineEntry[];
}

interface DateGroup {
  dateLabel: string;
  buckets:   BucketGroup[];
}

interface DwellRecord {
  zone_name:     string;
  kind:          string; // "camera" | "zone"
  dwell_seconds: number;
}

interface CameraPoint {
  camera_id:   string;
  camera_name: string | null;
  x:           number;
  y:           number;
  direction:   string;
  timestamp:   string;
}

interface HeatCount {
  gx: number;
  gy: number;
  count: number;
}

interface CameraOverlay {
  camera_id:      string;
  camera_name:    string;
  snapshotUrl:    string;
  naturalWidth:   number;
  naturalHeight:  number;
  points:         CameraPoint[];
  heatCounts:     HeatCount[];
  heatmapUrl:     string; // data URL PNG hasil accumulate→blur→normalize→colormap
  linePoints:     string; // atribut `points` SVG <polyline>, kosong kalau <2 titik
}

// Resolusi grid akumulasi (bukan resolusi tampilan — itu HEATMAP_CANVAS_PX).
// Cukup kasar; kehalusannya datang dari gaussian blur, bukan dari grid rapat.
const HEATMAP_GRID      = 24;
const HEATMAP_CANVAS_PX = 320; // sisi terpanjang kanvas output

type TabId = 'timeline' | 'pergerakan' | 'durasi';
type PergerakanView = 'garis' | 'heatmap';

@Component({
  selector: 'app-person-investigation',
  standalone: true,
  imports: [NgClass, AuthUrlPipe],
  templateUrl: './person-investigation.html',
  styleUrl: './person-investigation.css',
})
export class PersonInvestigation implements OnInit {
  private http   = inject(HttpClient);
  private route  = inject(ActivatedRoute);
  private router = inject(Router);
  private auth   = inject(AuthService);

  person: PersonDetail | null = null;
  dateGroups: DateGroup[] = [];
  loading = true;
  notFound = false;

  editMode    = false;
  editName    = '';
  editJabatan = '';

  activeTab: TabId = 'timeline';
  pergerakanView: PergerakanView = 'garis';
  dwell: DwellRecord[] = [];
  cameraOverlays: CameraOverlay[] = [];
  overlaysLoaded = false;
  dwellLoaded = false;

  // Modal footage — diputar di dalam halaman, bukan tab baru / download.
  footageUrl: string | null = null;

  ngOnInit(): void {
    const id = Number(this.route.snapshot.paramMap.get('id'));
    if (!id) { this.notFound = true; this.loading = false; return; }
    this._load(id);
  }

  private _load(id: number): void {
    this.http.get<PersonDetail>(`${API}/persons/${id}`).subscribe({
      next: person => {
        this.person  = person;
        this.loading = false;
        this._loadTimeline(id);
      },
      error: () => { this.notFound = true; this.loading = false; },
    });
  }

  private _loadTimeline(personId: number): void {
    const feedParams = new URLSearchParams({ person_id: String(personId), page: '1', limit: '100' });
    forkJoin({
      feed:      this.http.get<FeedResponse>(`${API}/people/feed?${feedParams}`),
      crossings: this.http.get<CrossingItem[]>(`${API}/persons/${personId}/crossings?limit=100`),
    }).subscribe({
      next: ({ feed, crossings }) => {
        // IN/OUT itu event tersendiri (crossing zona), bukan deteksi biasa —
        // digabung di sini cuma buat urutan waktu bersama, lalu ditampilkan
        // sebagai kartu terpisah (lihat template) supaya gak dikira kebetulan
        // berimpit dengan satu deteksi tertentu.
        const entries: TimelineEntry[] = [
          ...feed.items.map(d => ({
            kind: 'detection' as const, key: `d${d.detection_id}`, timestamp: d.timestamp, data: d,
          })),
          ...crossings.map(c => ({
            kind: 'event' as const, key: `e${c.id}`, timestamp: c.timestamp, data: c,
          })),
        ].sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
        this.dateGroups = this._groupByDateAndBucket(entries);
      },
      error: err => console.error('[person-investigation] timeline error:', err),
    });
  }

  private _groupByDateAndBucket(entries: TimelineEntry[]): DateGroup[] {
    const byDate = new Map<string, TimelineEntry[]>();
    for (const entry of entries) {
      const d = new Date(entry.timestamp);
      const dateKey = `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}`;
      if (!byDate.has(dateKey)) byDate.set(dateKey, []);
      byDate.get(dateKey)!.push(entry);
    }

    const groups: DateGroup[] = [];
    for (const dateItems of byDate.values()) {
      const buckets: BucketGroup[] = [];
      for (const entry of dateItems) {
        const label = this._bucketLabel(entry.timestamp);
        const last  = buckets[buckets.length - 1];
        if (last && last.label === label) {
          last.items.push(entry);
        } else {
          buckets.push({ label, items: [entry] });
        }
      }
      groups.push({ dateLabel: this._dateLabel(dateItems[0].timestamp), buckets });
    }
    return groups;
  }

  private _bucketLabel(iso: string): string {
    const d = new Date(iso);
    const startMin = Math.floor(d.getMinutes() / BUCKET_MINUTES) * BUCKET_MINUTES;
    const start = new Date(d); start.setMinutes(startMin, 0, 0);
    const end   = new Date(start.getTime() + BUCKET_MINUTES * 60_000);
    const fmt = (t: Date) => t.toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit' });
    return `${fmt(start)} – ${fmt(end)}`;
  }

  private _dateLabel(iso: string): string {
    return new Date(iso).toLocaleDateString('id-ID', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' });
  }

  fmtClock(iso: string): string {
    return new Date(iso).toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }

  fmtLastSeen(iso: string | null): string {
    if (!iso) return '—';
    return new Date(iso).toLocaleString('id-ID', { dateStyle: 'short', timeStyle: 'short' });
  }

  // ── Tabs ──────────────────────────────────────────────────────────────────

  selectTab(tab: TabId): void {
    this.activeTab = tab;
    if (!this.person) return;
    if (tab === 'pergerakan' && !this.overlaysLoaded) this._loadCameraOverlays(this.person.id);
    if (tab === 'durasi' && !this.dwellLoaded) this._loadDwell(this.person.id);
  }

  // Satu overlay per kamera yang punya titik posisi apa pun — dari crossing
  // zona (jarang) MAUPUN titik kaki tiap tracklet (tracklets.pos_x/pos_y,
  // hampir selalu ada, TIDAK butuh zona). `points` per kamera sudah urut
  // waktu (query backend ORDER BY timestamp), jadi bisa langsung dirangkai
  // jadi garis lintasan.
  private _loadCameraOverlays(personId: number): void {
    this.http.get<CameraPoint[]>(`${API}/persons/${personId}/camera-points`).subscribe({
      next: points => {
        const token = this.auth.getToken();
        const byCamera = new Map<string, CameraPoint[]>();
        for (const p of points) {
          if (!byCamera.has(p.camera_id)) byCamera.set(p.camera_id, []);
          byCamera.get(p.camera_id)!.push(p);
        }
        const qs = token ? `?token=${token}` : '';
        this.cameraOverlays = Array.from(byCamera.entries()).map(([camId, pts]) => ({
          camera_id:     camId,
          camera_name:   pts[0].camera_name || camId,
          snapshotUrl:   `${AI_API}/snapshot/${camId}${qs}`,
          naturalWidth:  0,
          naturalHeight: 0,
          points:        pts,
          heatCounts:    [],
          heatmapUrl:    '',
          linePoints:    '',
        }));
        this.overlaysLoaded = true;
      },
      error: err => console.error('[person-investigation] camera-points error:', err),
    });
  }

  onOverlayImgLoad(event: Event, overlay: CameraOverlay): void {
    const img = event.target as HTMLImageElement;
    overlay.naturalWidth  = img.naturalWidth;
    overlay.naturalHeight = img.naturalHeight;
    overlay.linePoints = this._computeLinePoints(overlay);
    overlay.heatCounts = this._computeHeatCounts(overlay);
    this._recomputeHeatColors();
  }

  // Rangkai titik-titik (urut waktu) jadi atribut `points` <polyline> SVG,
  // dalam ruang viewBox 0..100 (persentase).
  private _computeLinePoints(overlay: CameraOverlay): string {
    if (!overlay.naturalWidth || !overlay.naturalHeight || overlay.points.length < 2) return '';
    return overlay.points
      .map(p => `${(p.x / overlay.naturalWidth) * 100},${(p.y / overlay.naturalHeight) * 100}`)
      .join(' ');
  }

  // Langkah "Akumulasi posisi" dari pipeline CV standar: bin tiap titik ke
  // grid, hitung berapa kali tiap sel "kena". Blur & colormap-nya belum di
  // sini — itu tugas _renderHeatmap() (butuh max GLOBAL dulu, lihat di sana).
  private _computeHeatCounts(overlay: CameraOverlay): HeatCount[] {
    if (!overlay.naturalWidth || !overlay.naturalHeight || overlay.points.length === 0) return [];
    const counts = new Map<string, number>();
    for (const p of overlay.points) {
      const gx = Math.min(HEATMAP_GRID - 1, Math.floor((p.x / overlay.naturalWidth) * HEATMAP_GRID));
      const gy = Math.min(HEATMAP_GRID - 1, Math.floor((p.y / overlay.naturalHeight) * HEATMAP_GRID));
      const key = `${gx},${gy}`;
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    return Array.from(counts.entries()).map(([key, count]) => {
      const [gx, gy] = key.split(',').map(Number);
      return { gx, gy, count };
    });
  }

  // Pipeline heatmap CV standar, lewat <canvas>:
  //   akumulasi (sudah dari _computeHeatCounts) → gaussian blur (native
  //   canvas filter, bukan convolution manual) → normalisasi 0..1 →
  //   colormap biru→hijau→kuning→merah → PNG.
  // Normalisasi dihitung terhadap max GLOBAL (gabungan semua kamera orang
  // ini), bukan max per-kamera — kalau tiap kamera dinormalisasi sendiri,
  // orang yang cuma lewat 1x di satu kamera bikin sel itu langsung "merah
  // penuh" walau sebenarnya jarang, padahal kamera lain yang dia lewati
  // berkali-kali seharusnya yang paling merah.
  private _renderHeatmap(overlay: CameraOverlay, globalMax: number): void {
    if (!overlay.heatCounts.length || globalMax <= 0 || !overlay.naturalWidth) {
      overlay.heatmapUrl = '';
      return;
    }

    // 1) akumulasi → grid mentah grayscale (kecerahan = kepadatan mentah)
    const raw  = document.createElement('canvas');
    raw.width  = HEATMAP_GRID;
    raw.height = HEATMAP_GRID;
    const rctx = raw.getContext('2d')!;
    for (const { gx, gy, count } of overlay.heatCounts) {
      const v = Math.round((count / globalMax) * 255);
      rctx.fillStyle = `rgb(${v},${v},${v})`;
      rctx.fillRect(gx, gy, 1, 1);
    }

    // 2) upscale + gaussian blur — bikin area jadi smooth, bukan kotak-kotak
    const outW = HEATMAP_CANVAS_PX;
    const outH = Math.round(outW * (overlay.naturalHeight / overlay.naturalWidth));
    const blurred = document.createElement('canvas');
    blurred.width = outW; blurred.height = outH;
    const bctx = blurred.getContext('2d')!;
    bctx.filter = `blur(${outW / HEATMAP_GRID}px)`;
    bctx.drawImage(raw, 0, 0, outW, outH);

    // 3) normalisasi ulang (blur meratakan puncak) + mapping ke colormap,
    // per piksel — inilah "apply_colormap(heatmap)".
    const { data } = bctx.getImageData(0, 0, outW, outH);
    let peak = 1;
    for (let i = 0; i < data.length; i += 4) peak = Math.max(peak, data[i]);

    const out  = document.createElement('canvas');
    out.width  = outW; out.height = outH;
    const octx = out.getContext('2d')!;
    const img  = octx.createImageData(outW, outH);
    for (let i = 0; i < data.length; i += 4) {
      const t = data[i] / peak; // 0..1, kepadatan relatif final
      if (t < 0.03) { img.data[i + 3] = 0; continue; } // area kosong = transparan
      const [r, g, b] = this._colormap(t);
      img.data[i]     = r;
      img.data[i + 1] = g;
      img.data[i + 2] = b;
      img.data[i + 3] = Math.round((0.25 + 0.6 * t) * 255);
    }
    octx.putImageData(img, 0, 0);
    overlay.heatmapUrl = out.toDataURL();
  }

  // Biru(240°) → Hijau(120°) → Kuning(60°) → Merah(0°) sesuai kepadatan t.
  private _colormap(t: number): [number, number, number] {
    return this._hslToRgb(240 - 240 * t, 0.9, 0.5);
  }

  private _hslToRgb(h: number, s: number, l: number): [number, number, number] {
    h = ((h % 360) + 360) % 360;
    const c = (1 - Math.abs(2 * l - 1)) * s;
    const x = c * (1 - Math.abs((h / 60) % 2 - 1));
    const m = l - c / 2;
    let r = 0, g = 0, b = 0;
    if      (h < 60)  [r, g, b] = [c, x, 0];
    else if (h < 120) [r, g, b] = [x, c, 0];
    else if (h < 180) [r, g, b] = [0, c, x];
    else if (h < 240) [r, g, b] = [0, x, c];
    else if (h < 300) [r, g, b] = [x, 0, c];
    else              [r, g, b] = [c, 0, x];
    return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
  }

  private _recomputeHeatColors(): void {
    let globalMax = 0;
    for (const overlay of this.cameraOverlays) {
      for (const c of overlay.heatCounts) globalMax = Math.max(globalMax, c.count);
    }
    for (const overlay of this.cameraOverlays) this._renderHeatmap(overlay, globalMax);
  }

  private _loadDwell(personId: number): void {
    this.http.get<DwellRecord[]>(`${API}/persons/${personId}/dwell`).subscribe({
      next:  res => { this.dwell = res; this.dwellLoaded = true; },
      error: err => console.error('[person-investigation] dwell error:', err),
    });
  }

  fmtDuration(seconds: number): string {
    const h = Math.floor(seconds / 3600);
    const m = Math.round((seconds % 3600) / 60);
    if (h > 0) return `${h}j ${m}m`;
    if (seconds < 60) return `${Math.round(seconds)}d`;
    return `${m}m`;
  }

  openFootage(cameraId: string | null, timestamp: string, event?: Event): void {
    event?.stopPropagation();
    if (!cameraId) return;
    const token  = this.auth.getToken();
    const params = new URLSearchParams({ timestamp });
    if (token) params.set('token', token);
    this.footageUrl = `${AI_API}/clips/${cameraId}?${params}`;
  }

  closeFootage(): void {
    this.footageUrl = null;
  }

  // ── Ubah nama ─────────────────────────────────────────────────────────────

  openEdit(): void {
    if (!this.person) return;
    this.editName    = this.person.name;
    this.editJabatan = this.person.jabatan ?? '';
    this.editMode    = true;
  }

  cancelEdit(): void { this.editMode = false; }

  saveEdit(): void {
    if (!this.person) return;
    const name = this.editName.trim();
    if (!name) return;

    this.http.patch<PersonDetail>(`${API}/persons/${this.person.id}`, {
      name,
      jabatan: this.editJabatan.trim() || null,
    }).subscribe({
      next: updated => {
        this.editMode = false;
        // Nama yang sama persis dengan orang lain yang sudah dikenal → backend
        // menggabungkan keduanya dan mengembalikan id ORANG TARGET (bukan id
        // halaman ini) — pindah ke sana supaya URL & datanya konsisten.
        if (updated.id !== this.person!.id) {
          this.router.navigate(['/people', updated.id]);
          return;
        }
        this.person!.name     = updated.name;
        this.person!.jabatan  = updated.jabatan;
        this.person!.is_known = updated.is_known;
      },
      error: err => console.error('[person-investigation] rename error:', err),
    });
  }

  goBack(): void {
    this.router.navigate(['/people']);
  }
}

import { Component, OnInit, inject } from '@angular/core';
import { NgClass, NgStyle } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { ActivatedRoute, Router } from '@angular/router';
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
  detection_id:  number;
  person_id:     number | null;
  camera_id:     string;
  camera_name:   string | null;
  timestamp:     string;
  thumbnail_url: string | null;
  tracklet_id:   number | null;
}

interface FeedResponse {
  items: FeedItem[];
  total: number;
  page:  number;
  pages: number;
}

interface BucketGroup {
  label: string;
  items: FeedItem[];
}

interface DateGroup {
  dateLabel: string;
  buckets:   BucketGroup[];
}

interface DwellRecord {
  zone_name:     string;
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

interface HeatCell {
  left: string;
  top:  string;
  size: string;
  opacity: number;
}

interface CameraOverlay {
  camera_id:      string;
  camera_name:    string;
  snapshotUrl:    string;
  naturalWidth:   number;
  naturalHeight:  number;
  points:         CameraPoint[];
  heatCells:      HeatCell[];
  linePoints:     string; // atribut `points` SVG <polyline>, kosong kalau <2 titik
}

const HEATMAP_GRID = 8; // ponytail: grid tetap 8x8, cukup buat jumlah titik per orang yang biasanya sedikit

type TabId = 'timeline' | 'pergerakan' | 'durasi';
type PergerakanView = 'garis' | 'heatmap';

@Component({
  selector: 'app-person-investigation',
  standalone: true,
  imports: [NgClass, NgStyle, AuthUrlPipe],
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
    const params = new URLSearchParams({ person_id: String(personId), page: '1', limit: '100' });
    this.http.get<FeedResponse>(`${API}/people/feed?${params}`).subscribe({
      next:  res => this.dateGroups = this._groupByDateAndBucket(res.items),
      error: err => console.error('[person-investigation] timeline error:', err),
    });
  }

  private _groupByDateAndBucket(items: FeedItem[]): DateGroup[] {
    const byDate = new Map<string, FeedItem[]>();
    for (const item of items) {
      const d = new Date(item.timestamp);
      const dateKey = `${d.getFullYear()}-${d.getMonth()}-${d.getDate()}`;
      if (!byDate.has(dateKey)) byDate.set(dateKey, []);
      byDate.get(dateKey)!.push(item);
    }

    const groups: DateGroup[] = [];
    for (const dateItems of byDate.values()) {
      const buckets: BucketGroup[] = [];
      for (const item of dateItems) {
        const label = this._bucketLabel(item.timestamp);
        const last  = buckets[buckets.length - 1];
        if (last && last.label === label) {
          last.items.push(item);
        } else {
          buckets.push({ label, items: [item] });
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
          heatCells:     [],
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
    overlay.heatCells  = this._computeHeatCells(overlay);
    overlay.linePoints = this._computeLinePoints(overlay);
  }

  dotStyle(overlay: CameraOverlay, point: CameraPoint): Record<string, string> {
    if (!overlay.naturalWidth || !overlay.naturalHeight) return { display: 'none' };
    return {
      left: `${(point.x / overlay.naturalWidth) * 100}%`,
      top:  `${(point.y / overlay.naturalHeight) * 100}%`,
    };
  }

  dotClass(point: CameraPoint): string {
    if (point.direction === 'IN')  return 'overlay-dot--in';
    if (point.direction === 'OUT') return 'overlay-dot--out';
    return 'overlay-dot--track';
  }

  // Rangkai titik-titik (urut waktu) jadi atribut `points` <polyline> SVG,
  // dalam ruang viewBox 0..100 (persentase) — sinkron dengan dotStyle().
  private _computeLinePoints(overlay: CameraOverlay): string {
    if (!overlay.naturalWidth || !overlay.naturalHeight || overlay.points.length < 2) return '';
    return overlay.points
      .map(p => `${(p.x / overlay.naturalWidth) * 100},${(p.y / overlay.naturalHeight) * 100}`)
      .join(' ');
  }

  // Grid kepadatan sederhana dari titik-titik lintas yang ada — bukan
  // heatmap Gaussian, cuma binning per sel supaya area yang sering dilewati
  // kelihatan lebih "panas".
  private _computeHeatCells(overlay: CameraOverlay): HeatCell[] {
    if (!overlay.naturalWidth || !overlay.naturalHeight || overlay.points.length === 0) return [];
    const counts = new Map<string, number>();
    for (const p of overlay.points) {
      const gx = Math.min(HEATMAP_GRID - 1, Math.floor((p.x / overlay.naturalWidth) * HEATMAP_GRID));
      const gy = Math.min(HEATMAP_GRID - 1, Math.floor((p.y / overlay.naturalHeight) * HEATMAP_GRID));
      const key = `${gx},${gy}`;
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    const max = Math.max(...counts.values());
    const cellPct = 100 / HEATMAP_GRID;
    return Array.from(counts.entries()).map(([key, count]) => {
      const [gx, gy] = key.split(',').map(Number);
      return {
        left: `${gx * cellPct}%`,
        top:  `${gy * cellPct}%`,
        size: `${cellPct}%`,
        opacity: 0.25 + 0.55 * (count / max),
      };
    });
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
    return `${m}m`;
  }

  openFootage(item: FeedItem, event?: Event): void {
    event?.stopPropagation();
    const token  = this.auth.getToken();
    const params = new URLSearchParams({ timestamp: item.timestamp });
    if (token) params.set('token', token);
    this.footageUrl = `${AI_API}/clips/${item.camera_id}?${params}`;
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

import {
  Component,
  ElementRef,
  OnInit,
  QueryList,
  ViewChild,
  ViewChildren,
} from '@angular/core';
import { NgClass, SlicePipe } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { forkJoin } from 'rxjs';
import { AuthUrlPipe } from '../../pipes/auth-url.pipe';
import { environment } from '../../../environments/environment';

const API = environment.apiBaseUrl;
const AI  = environment.aiServiceBaseUrl;

type ZoneType = 'line' | 'polygon';

interface Point { x: number; y: number; }
interface LineSegment { p1: Point; p2: Point; in_sign: number; }

interface Zone {
  id:            number;
  name:          string;
  max_capacity:  number | null;
  camera_count:  number;
  camera_ids:    string[];
  types:         ZoneType[];   // tipe gambar unik yang ada di zona ini (bisa campur)
  created_at:    string;
}

interface ZoneCamera {
  id:            number;   // zone_camera_id
  zone_id:       number;
  camera_id:     number;
  camera_str_id: string | null;
  camera_name:   string | null;
  type:          ZoneType;
  points:        LineSegment[] | Point[];
}

interface ZoneDetail extends Zone {
  cameras: ZoneCamera[];
}

interface Camera {
  id:         number;
  camera_id:  string | null;
  name:       string;
  location:   string | null;
}

interface OccupancyRow {
  zone_id:            number;
  zone_name:          string;
  max_capacity:       number | null;
  count_in:           number;
  count_out:          number;
  current_occupancy:  number;
}

interface HistoryPoint { date: string; count_in: number; count_out: number; }
interface Heatmap { grid_size: number; cells: number[][]; }

interface ZoneEvent {
  id:           number;
  timestamp:    string;
  direction:    string;
  person_label: string | null;
  snapshot_url: string | null;
  camera_id:    string | null;
  camera_name:  string | null;
}

type RangeMode = 'today' | '7days' | 'custom';

@Component({
  selector: 'app-zone',
  standalone: true,
  imports: [NgClass, SlicePipe, AuthUrlPipe],
  templateUrl: './zone.html',
  styleUrl: './zone.css',
})
export class ZonePage implements OnInit {
  @ViewChildren('viewSnapImg') viewSnapImgs!: QueryList<ElementRef<HTMLImageElement>>;
  @ViewChild('configSnapImg') configSnapImg!: ElementRef<HTMLImageElement>;
  @ViewChild('configCanvas')  configCanvas!: ElementRef<HTMLCanvasElement>;

  zones:      Zone[]          = [];
  cameras:    Camera[]        = [];
  occupancy:  OccupancyRow[]  = [];

  // Search & filter
  searchQuery: string = '';
  typeFilter:  'semua' | ZoneType = 'semua';
  cameraFilter: string = 'semua';

  // Create zone modal
  showCreateModal = false;
  createName       = '';
  createCameraId   = '';
  createCapacity: number | null = null;
  isCreatingNewZone = false;   // true = drawingCamera lagi dipakai buat bikin zona baru, bukan edit yang lama

  constructor(private http: HttpClient) {}

  ngOnInit(): void {
    this.loadAll();
  }

  loadAll(): void {
    forkJoin({
      zones:     this.http.get<Zone[]>(`${API}/zones`),
      cameras:   this.http.get<Camera[]>(`${API}/cameras`),
      occupancy: this.http.get<OccupancyRow[]>(`${API}/occupancy`),
    }).subscribe(({ zones, cameras, occupancy }) => {
      this.zones     = zones;
      this.cameras   = cameras;
      this.occupancy = occupancy;
    });
  }

  occupancyOf(zoneId: number): OccupancyRow | null {
    return this.occupancy.find(o => o.zone_id === zoneId) ?? null;
  }

  // ── Search & filter ──────────────────────────────────────────────────────

  onSearchInput(event: Event): void {
    this.searchQuery = (event.target as HTMLInputElement).value;
  }

  onTypeFilterChange(event: Event): void {
    this.typeFilter = (event.target as HTMLSelectElement).value as 'semua' | ZoneType;
  }

  onCameraFilterChange(event: Event): void {
    this.cameraFilter = (event.target as HTMLSelectElement).value;
  }

  get filteredZones(): Zone[] {
    const q = this.searchQuery.trim().toLowerCase();
    return this.zones.filter(z => {
      const matchSearch = !q || z.name.toLowerCase().includes(q);
      const matchType   = this.typeFilter === 'semua' || z.types.includes(this.typeFilter as ZoneType);
      const matchCamera = this.cameraFilter === 'semua' || z.camera_ids.includes(this.cameraFilter);
      return matchSearch && matchType && matchCamera;
    });
  }

  // ── Create zone ───────────────────────────────────────────────────────────
  // Zona wajib langsung punya minimal 1 sumber kamera — modal ini cuma nampung
  // nama/kapasitas/pilih-kamera, lalu lanjut ke panel gambar (drawingCamera)
  // yang sama dipakai buat tambah/edit kamera di zona existing.

  openCreateModal(): void {
    this.createName       = '';
    this.createCapacity   = null;
    this.createCameraId   = this.cameras.find(c => c.camera_id)?.camera_id ?? '';
    this.showCreateModal  = true;
  }

  closeCreateModal(): void { this.showCreateModal = false; }

  onCreateCameraChange(event: Event): void {
    this.createCameraId = (event.target as HTMLSelectElement).value;
  }

  proceedToCreateDrawing(): void {
    const name = this.createName.trim();
    if (!name || !this.createCameraId) return;
    this.showCreateModal   = false;
    this.isCreatingNewZone = true;
    const cam = this.cameras.find(c => c.camera_id === this.createCameraId);
    this.drawingCamera = {
      id: 0, zone_id: 0, camera_id: 0,
      camera_str_id: this.createCameraId, camera_name: cam?.name ?? '', type: 'line', points: [],
    };
    this.drawingCameraStrId = this.createCameraId;
    this.setDrawingType('line');
    this._resetDrawingState();
    this._loadDrawingSnapshot();
  }

  // ══════════════════════════════════════════════════════════════════════════
  // ── VIEW modal ───────────────────────────────────────────────────────────
  // ══════════════════════════════════════════════════════════════════════════

  showViewModal = false;
  viewZone: ZoneDetail | null = null;

  rangeMode: RangeMode = 'today';
  customFrom = '';
  customTo   = '';

  historyPoints: HistoryPoint[] = [];

  heatmapCameraId: string | null = null;
  heatmapSnapshotUrl = '';
  viewCamSnapshotUrls: Record<string, string> = {};   // camera_str_id -> URL, diambil sekali pas modal dibuka
  heatmap: Heatmap | null = null;

  events: ZoneEvent[] = [];
  eventsPage  = 1;
  eventsPages = 1;
  eventsTotal = 0;

  playingClip: ZoneEvent | null = null;
  playingClipUrl = '';

  private _todayISO(): string {
    const d = new Date();
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
  }

  private _rangeDates(): { from: string; to: string } {
    const to = this._todayISO();
    if (this.rangeMode === 'today') return { from: to, to };
    if (this.rangeMode === '7days') {
      const d = new Date();
      d.setDate(d.getDate() - 6);
      const from = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
      return { from, to };
    }
    return { from: this.customFrom || to, to: this.customTo || to };
  }

  openViewModal(zone: Zone, event: Event): void {
    event.stopPropagation();
    this.http.get<ZoneDetail>(`${API}/zones/${zone.id}`).subscribe(detail => {
      this.viewZone   = detail;
      this.showViewModal = true;
      this.rangeMode  = 'today';
      this.heatmapCameraId = detail.cameras[0]?.camera_str_id ?? null;
      this.eventsPage = 1;
      // Snapshot diambil sekali pas modal dibuka — bukan tiap change-detection
      // cycle, biar gambarnya gak reload/kedip terus.
      this.viewCamSnapshotUrls = {};
      for (const cam of detail.cameras) {
        if (cam.camera_str_id) {
          this.viewCamSnapshotUrls[cam.camera_str_id] = `${API}/cameras/${cam.camera_str_id}/snapshot?t=${Date.now()}`;
        }
      }
      this.heatmapSnapshotUrl = this.heatmapCameraId ? this.viewCamSnapshotUrls[this.heatmapCameraId] ?? '' : '';
      this._reloadViewData();
    });
  }

  closeViewModal(): void {
    this.showViewModal = false;
    this.viewZone = null;
    this.playingClip = null;
  }

  setRangeMode(mode: RangeMode): void {
    this.rangeMode = mode;
    this._reloadViewData();
  }

  onCustomFromChange(event: Event): void {
    this.customFrom = (event.target as HTMLInputElement).value;
    if (this.rangeMode === 'custom') this._reloadViewData();
  }

  onCustomToChange(event: Event): void {
    this.customTo = (event.target as HTMLInputElement).value;
    if (this.rangeMode === 'custom') this._reloadViewData();
  }

  onHeatmapCameraChange(event: Event): void {
    this.heatmapCameraId = (event.target as HTMLSelectElement).value;
    this.heatmapSnapshotUrl = this.viewCamSnapshotUrls[this.heatmapCameraId] ?? '';
    this._loadHeatmap();
  }

  private _reloadViewData(): void {
    if (!this.viewZone) return;
    const { from, to } = this._rangeDates();
    this.http.get<HistoryPoint[]>(`${API}/zones/${this.viewZone.id}/history?from=${from}&to=${to}`)
      .subscribe(points => { this.historyPoints = points; });
    this._loadHeatmap();
    this._loadEvents();
  }

  private _loadHeatmap(): void {
    if (!this.viewZone || !this.heatmapCameraId) { this.heatmap = null; return; }
    const { from, to } = this._rangeDates();
    this.http.get<Heatmap>(
      `${API}/zones/${this.viewZone.id}/heatmap?camera_id=${this.heatmapCameraId}&from=${from}&to=${to}`
    ).subscribe(h => { this.heatmap = h; });
  }

  loadEventsPage(page: number): void {
    this.eventsPage = page;
    this._loadEvents();
  }

  private _loadEvents(): void {
    if (!this.viewZone) return;
    const { from, to } = this._rangeDates();
    this.http.get<{ events: ZoneEvent[]; total: number; pages: number }>(
      `${API}/zones/${this.viewZone.id}/events?from=${from}&to=${to}&page=${this.eventsPage}&limit=10`
    ).subscribe(res => {
      this.events      = res.events;
      this.eventsTotal = res.total;
      this.eventsPages = res.pages;
    });
  }

  playClip(ev: ZoneEvent): void {
    if (!ev.camera_id) return;
    this.playingClip    = ev;
    this.playingClipUrl = `${AI}/clips/${ev.camera_id}?timestamp=${encodeURIComponent(ev.timestamp)}`;
  }

  closeClipPlayer(): void {
    this.playingClip = null;
  }

  get eventsPageNumbers(): number[] {
    return Array.from({ length: this.eventsPages }, (_, i) => i + 1);
  }

  get historyMax(): number {
    return Math.max(1, ...this.historyPoints.map(p => Math.max(p.count_in, p.count_out)));
  }

  get heatmapMax(): number {
    if (!this.heatmap) return 1;
    return Math.max(1, ...this.heatmap.cells.flat());
  }

  heatmapOpacity(count: number): number {
    if (count === 0) return 0;
    return 0.15 + (count / this.heatmapMax) * 0.65;
  }

  onViewSnapshotLoad(index: number): void {
    const imgEl = this.viewSnapImgs.get(index)?.nativeElement;
    if (!imgEl) return;
    const canvas = document.getElementById(`view-overlay-${index}`) as HTMLCanvasElement | null;
    if (!canvas || !this.viewZone) return;
    canvas.width  = imgEl.naturalWidth  || imgEl.offsetWidth;
    canvas.height = imgEl.naturalHeight || imgEl.offsetHeight;
    const cam = this.viewZone.cameras[index];
    this._drawZoneShape(canvas, cam, 'rgba(231,0,11,0.85)');
  }

  private _drawZoneShape(canvas: HTMLCanvasElement, cam: ZoneCamera, color: string): void {
    const ctx = canvas.getContext('2d');
    if (!ctx || !this.viewZone) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = color;
    ctx.fillStyle   = color;
    ctx.lineWidth   = Math.max(3, canvas.width / 300);

    if (cam.type === 'line') {
      const lw       = Math.max(3, canvas.width / 300);
      const fontSize = Math.max(18, canvas.width / 40);
      for (const seg of cam.points as LineSegment[]) {
        ctx.beginPath();
        ctx.moveTo(seg.p1.x, seg.p1.y);
        ctx.lineTo(seg.p2.x, seg.p2.y);
        ctx.stroke();
        this._drawDirectionArrow(ctx, seg.p1.x, seg.p1.y, seg.p2.x, seg.p2.y, seg.in_sign, color, lw, fontSize);
      }
    } else {
      const pts = cam.points as Point[];
      if (pts.length < 2) return;
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (const p of pts.slice(1)) ctx.lineTo(p.x, p.y);
      ctx.closePath();
      ctx.globalAlpha = 0.2;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.stroke();
    }
  }

  // ══════════════════════════════════════════════════════════════════════════
  // ── KONFIGURASI modal ────────────────────────────────────────────────────
  // ══════════════════════════════════════════════════════════════════════════

  showConfigModal = false;
  configZone: ZoneDetail | null = null;
  configName     = '';
  configCapacity: number | null = null;

  // Tambah kamera baru ke zona
  configAddCameraId: string = '';

  // Drawing sub-panel — kamera yang lagi diedit geometrinya. Tipe (line/polygon)
  // ditanyakan di sini, per gambar — bukan atribut zona.
  drawingCamera: ZoneCamera | null = null;
  drawingCameraStrId = '';
  drawingType: ZoneType = 'line';
  drawingSnapshotUrl = '';
  drawingSnapshotError = false;
  savedSegments: LineSegment[] = [];      // drawingType='line'
  polygonPoints: Point[]       = [];       // drawingType='polygon', in-progress atau final
  private _clickBuffer: Point[] = [];      // buffer 2 klik utk 1 segmen garis

  openConfigModal(zone: Zone, event: Event): void {
    event.stopPropagation();
    this.http.get<ZoneDetail>(`${API}/zones/${zone.id}`).subscribe(detail => {
      this.configZone     = detail;
      this.configName     = detail.name;
      this.configCapacity = detail.max_capacity;
      this.configAddCameraId = '';
      this.drawingCamera  = null;
      this.showConfigModal = true;
    });
  }

  closeConfigModal(): void {
    this.showConfigModal = false;
    this.configZone = null;
    this.drawingCamera = null;
  }

  saveZoneMeta(): void {
    if (!this.configZone) return;
    const name = this.configName.trim();
    if (!name) return;
    this.http.put<Zone>(`${API}/zones/${this.configZone.id}`, {
      name, max_capacity: this.configCapacity,
    }).subscribe(() => {
      this.loadAll();
      this.closeConfigModal();
    });
  }

  deleteZone(): void {
    if (!this.configZone) return;
    if (!confirm(`Hapus zona "${this.configZone.name}"?`)) return;
    this.http.delete(`${API}/zones/${this.configZone.id}`).subscribe(() => {
      this.loadAll();
      this.closeConfigModal();
    });
  }

  get availableCamerasToAdd(): Camera[] {
    if (!this.configZone) return [];
    const assigned = new Set(this.configZone.cameras.map(c => c.camera_str_id));
    return this.cameras.filter(c => c.camera_id && !assigned.has(c.camera_id));
  }

  onConfigAddCameraChange(event: Event): void {
    this.configAddCameraId = (event.target as HTMLSelectElement).value;
  }

  addCameraToZone(): void {
    if (!this.configZone || !this.configAddCameraId) return;
    const cam = this.cameras.find(c => c.camera_id === this.configAddCameraId);
    if (!cam) return;
    // Placeholder kosong dulu — geometri digambar & disimpan lewat panel drawing
    this.startDrawingForNewCamera(this.configAddCameraId, cam.name);
  }

  removeCameraFromZone(cam: ZoneCamera, event: Event): void {
    event.stopPropagation();
    if (!this.configZone || !cam.camera_str_id) return;
    if (!confirm(`Lepas kamera "${cam.camera_name}" dari zona ini?`)) return;
    this.http.delete(`${API}/zones/${this.configZone.id}/cameras/${cam.camera_str_id}`).subscribe({
      next: () => this.openConfigModal({ id: this.configZone!.id } as Zone, new Event('click')),
      error: (err) => alert(err.error?.detail ?? 'Gagal melepas kamera.'),
    });
  }

  // ── Drawing sub-panel ────────────────────────────────────────────────────

  startDrawingCamera(cam: ZoneCamera, event: Event): void {
    event.stopPropagation();
    this.isCreatingNewZone   = false;
    this.drawingCamera       = cam;
    this.drawingCameraStrId  = cam.camera_str_id ?? '';
    this._resetDrawingState();
    this.drawingType = cam.type;
    if (cam.type === 'line') {
      this.savedSegments = [...(cam.points as LineSegment[])];
    } else {
      this.polygonPoints = [...(cam.points as Point[])];
    }
    this._loadDrawingSnapshot();
  }

  startDrawingForNewCamera(camStrId: string, camName: string): void {
    this.isCreatingNewZone = false;
    this.drawingCamera = {
      id: 0, zone_id: this.configZone?.id ?? 0, camera_id: 0,
      camera_str_id: camStrId, camera_name: camName, type: 'line', points: [],
    };
    this.drawingCameraStrId = camStrId;
    this.setDrawingType('line');
    this._resetDrawingState();
    this._loadDrawingSnapshot();
  }

  setDrawingType(type: ZoneType): void {
    if (this.drawingType === type) return;
    this.drawingType = type;
    this._resetDrawingState();
    this._drawConfigCanvas();
  }

  private _resetDrawingState(): void {
    this.savedSegments = [];
    this.polygonPoints = [];
    this._clickBuffer  = [];
    this.drawingSnapshotError = false;
  }

  cancelDrawing(): void {
    this.drawingCamera = null;
    this.isCreatingNewZone = false;
  }

  private _loadDrawingSnapshot(): void {
    this.drawingSnapshotUrl   = `${API}/cameras/${this.drawingCameraStrId}/snapshot?t=${Date.now()}`;
    this.drawingSnapshotError = false;
  }

  refreshDrawingSnapshot(): void {
    this._loadDrawingSnapshot();
  }

  onDrawingSnapshotLoad(): void {
    this.drawingSnapshotError = false;
    const img    = this.configSnapImg.nativeElement;
    const canvas = this.configCanvas.nativeElement;
    canvas.width  = img.naturalWidth  || img.offsetWidth;
    canvas.height = img.naturalHeight || img.offsetHeight;
    this._drawConfigCanvas();
  }

  onDrawingSnapshotError(): void {
    this.drawingSnapshotError = true;
  }

  onConfigCanvasClick(event: MouseEvent): void {
    const canvas = this.configCanvas.nativeElement;
    const img    = this.configSnapImg.nativeElement;
    if (!img.naturalWidth) return;

    const rect   = canvas.getBoundingClientRect();
    const scaleX = img.naturalWidth  / rect.width;
    const scaleY = img.naturalHeight / rect.height;
    const x = Math.round((event.clientX - rect.left) * scaleX);
    const y = Math.round((event.clientY - rect.top)  * scaleY);

    if (this.drawingType === 'line') {
      this._clickBuffer.push({ x, y });
      if (this._clickBuffer.length >= 2) {
        this.savedSegments.push({ p1: this._clickBuffer[0], p2: this._clickBuffer[1], in_sign: 1 });
        this._clickBuffer = [];
      }
    } else {
      this.polygonPoints.push({ x, y });
    }
    this._drawConfigCanvas();
  }

  flipSegmentSign(index: number): void {
    this.savedSegments[index].in_sign *= -1;
    this._drawConfigCanvas();
  }

  removeSegment(index: number): void {
    this.savedSegments.splice(index, 1);
    this._drawConfigCanvas();
  }

  clearPolygon(): void {
    this.polygonPoints = [];
    this._drawConfigCanvas();
  }

  undoLastPolygonPoint(): void {
    this.polygonPoints.pop();
    this._drawConfigCanvas();
  }

  private _drawConfigCanvas(): void {
    const canvas = this.configCanvas?.nativeElement;
    const img    = this.configSnapImg?.nativeElement;
    if (!canvas || !img?.naturalWidth) return;
    const ctx = canvas.getContext('2d')!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const lw       = Math.max(4, img.naturalWidth / 300);
    const dotR     = lw * 2.5;
    const fontSize = Math.max(18, img.naturalWidth / 40);

    if (this.drawingType === 'line') {
      this.savedSegments.forEach(seg => {
        ctx.strokeStyle = '#E7000B';
        ctx.lineWidth   = lw;
        ctx.beginPath();
        ctx.moveTo(seg.p1.x, seg.p1.y);
        ctx.lineTo(seg.p2.x, seg.p2.y);
        ctx.stroke();
        ctx.fillStyle = '#E7000B';
        [seg.p1, seg.p2].forEach(p => {
          ctx.beginPath(); ctx.arc(p.x, p.y, dotR, 0, Math.PI * 2); ctx.fill();
        });
        this._drawDirectionArrow(
          ctx, seg.p1.x, seg.p1.y, seg.p2.x, seg.p2.y,
          seg.in_sign, '#E7000B', lw, fontSize,
        );
      });
      if (this._clickBuffer[0]) {
        ctx.fillStyle = 'rgba(255,255,255,0.9)';
        ctx.beginPath(); ctx.arc(this._clickBuffer[0].x, this._clickBuffer[0].y, dotR, 0, Math.PI * 2); ctx.fill();
      }
    } else {
      if (this.polygonPoints.length > 0) {
        ctx.strokeStyle = '#E7000B';
        ctx.fillStyle   = 'rgba(231,0,11,0.15)';
        ctx.lineWidth   = lw;
        ctx.beginPath();
        ctx.moveTo(this.polygonPoints[0].x, this.polygonPoints[0].y);
        for (const p of this.polygonPoints.slice(1)) ctx.lineTo(p.x, p.y);
        if (this.polygonPoints.length >= 3) { ctx.closePath(); ctx.fill(); }
        ctx.stroke();
        ctx.fillStyle = '#E7000B';
        this.polygonPoints.forEach(p => {
          ctx.beginPath(); ctx.arc(p.x, p.y, dotR, 0, Math.PI * 2); ctx.fill();
        });
      }
    }
  }

  /** Panah tegak lurus dari tengah garis menunjuk ke sisi IN, biar arah jelas
   * pas konfigurasi — bukan cuma nomor "Garis 1" tanpa gambaran arah. */
  private _drawDirectionArrow(
    ctx: CanvasRenderingContext2D,
    x1: number, y1: number, x2: number, y2: number,
    inSign: number,
    color: string, lw: number, fontSize: number,
  ): void {
    const dx  = x2 - x1;
    const dy  = y2 - y1;
    const len = Math.sqrt(dx * dx + dy * dy);
    if (len === 0) return;

    const perpX = (-dy / len) * inSign;
    const perpY = ( dx / len) * inSign;

    const mx = (x1 + x2) / 2;
    const my = (y1 + y2) / 2;
    const arrowLen = Math.max(40, len * 0.25);
    const headLen  = arrowLen * 0.35;
    const ax = mx + perpX * arrowLen;
    const ay = my + perpY * arrowLen;
    const angle = Math.atan2(perpY, perpX);

    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle   = color;
    ctx.lineWidth   = lw * 0.8;
    ctx.shadowBlur  = 0;

    ctx.beginPath();
    ctx.moveTo(mx, my);
    ctx.lineTo(ax, ay);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(ax - headLen * Math.cos(angle - Math.PI / 6), ay - headLen * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(ax - headLen * Math.cos(angle + Math.PI / 6), ay - headLen * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();

    ctx.font         = `bold ${fontSize}px sans-serif`;
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'middle';
    ctx.strokeStyle  = 'rgba(0,0,0,0.6)';
    ctx.lineWidth    = fontSize * 0.15;
    const lx = ax + perpX * (fontSize * 0.9);
    const ly = ay + perpY * (fontSize * 0.9);
    ctx.strokeText('IN', lx, ly);
    ctx.fillText('IN', lx, ly);

    ctx.restore();
  }

  saveDrawing(): void {
    if (!this.drawingCamera || !this.drawingCameraStrId) return;
    const points = this.drawingType === 'line' ? this.savedSegments : this.polygonPoints;
    if (this.drawingType === 'polygon' && points.length < 3) {
      alert('Polygon butuh minimal 3 titik.');
      return;
    }
    if (this.drawingType === 'line' && points.length === 0) {
      alert('Gambar minimal 1 garis dulu.');
      return;
    }

    if (this.isCreatingNewZone) {
      const name = this.createName.trim();
      this.http.post<Zone>(`${API}/zones`, {
        name, max_capacity: this.createCapacity,
        camera_id: this.drawingCameraStrId, type: this.drawingType, points,
      }).subscribe(() => {
        this.isCreatingNewZone = false;
        this.drawingCamera = null;
        this.loadAll();
      });
      return;
    }

    if (!this.configZone) return;
    this.http.put(`${API}/zones/${this.configZone.id}/cameras/${this.drawingCameraStrId}`, {
      type: this.drawingType, points,
    }).subscribe(() => {
      const zoneId = this.configZone!.id;
      this.drawingCamera = null;
      this.http.get<ZoneDetail>(`${API}/zones/${zoneId}`).subscribe(detail => {
        this.configZone = detail;
      });
      this.loadAll();
    });
  }
}

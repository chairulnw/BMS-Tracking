import {
  AfterViewInit,
  Component,
  DestroyRef,
  ElementRef,
  inject,
  OnInit,
  ViewChild,
} from '@angular/core';
import { DecimalPipe, NgClass } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { interval } from 'rxjs';
import { startWith, switchMap } from 'rxjs/operators';
import { AuthUrlPipe } from '../../pipes/auth-url.pipe';

const API = 'http://localhost:8002';
const AI  = 'http://localhost:8001';

const LINE_COLORS = ['#ef4444', '#3b82f6', '#22c55e', '#f59e0b', '#a855f7'];

interface Camera {
  id:            number;
  camera_id:     string | null;
  name:          string;
  zone_location: string | null;
  floor:         string | null;
  rtsp_url:      string;
  is_active:     boolean;
}

interface CrossingLine {
  id:        number;
  camera_id: string;
  p1_x:      number;
  p1_y:      number;
  p2_x:      number;
  p2_y:      number;
  in_sign:   number;
}

interface Point { x: number; y: number; }

@Component({
  selector: 'app-pengaturan',
  standalone: true,
  imports: [NgClass, DecimalPipe, AuthUrlPipe],
  templateUrl: './pengaturan.html',
  styleUrl: './pengaturan.css',
})
export class Pengaturan implements OnInit, AfterViewInit {
  @ViewChild('snapshotImg') imgRef!: ElementRef<HTMLImageElement>;
  @ViewChild('zoneCanvas')  canvasRef!: ElementRef<HTMLCanvasElement>;

  cameras:  Camera[]  = [];
  selectedCamera: Camera | null = null;

  // Zone canvas
  snapshotUrl  = '';
  snapshotError = false;
  savedLines:  CrossingLine[] = [];
  points:      Point[]        = [];      // 0 or 1 interim click points
  pendingLine: { p1: Point; p2: Point; inSign: number } | null = null;

  // Detail Zona form
  showZoneForm = false;
  roomName     = '';

  // Camera modal
  showModal       = false;
  editingCamera: Camera | null = null;
  formName        = '';
  formFloor       = '';
  formRtspUrl     = '';
  formIsActive    = true;

  // RTSP show/hide per row
  visibleRtsp = new Set<number>();

  // Camera table filters
  floorFilter:  string = 'semua';
  statusFilter: 'semua' | 'aktif' | 'nonaktif' = 'semua';

  // Stream state
  streamRunning  = false;
  streamLoading  = false;
  framesProcessed = 0;

  private destroyRef = inject(DestroyRef);

  constructor(private http: HttpClient) {}

  ngOnInit(): void {
    this.loadCameras();
    this._pollStreamStatus();
  }

  ngAfterViewInit(): void {}

  private _pollStreamStatus(): void {
    interval(3000).pipe(
      startWith(0),
      switchMap(() => this.http.get<{ running: boolean; frames_processed: number }>(
        `${AI}/stream/status`
      )),
      takeUntilDestroyed(this.destroyRef),
    ).subscribe({
      next: s => {
        this.streamRunning   = s.running;
        this.framesProcessed = s.frames_processed;
      },
      error: () => { this.streamRunning = false; },
    });
  }

  startStream(): void {
    if (this.streamLoading) return;
    this.streamLoading = true;
    this.http.post(`${AI}/stream/start`, {}).subscribe({
      next: () => { this.streamRunning = true;  this.streamLoading = false; },
      error: ()=> { this.streamLoading = false; },
    });
  }

  stopStream(): void {
    if (this.streamLoading) return;
    this.streamLoading = true;
    this.http.post(`${AI}/stream/stop`, {}).subscribe({
      next: () => { this.streamRunning = false; this.streamLoading = false; },
      error: ()=> { this.streamLoading = false; },
    });
  }

  // ── Camera list ───────────────────────────────────────────────────────────

  loadCameras(): void {
    this.http.get<Camera[]>(`${API}/cameras`).subscribe({
      next:  cams => {
        this.cameras = cams;
        if (!this.selectedCamera && cams.length) {
          this.selectCamera(cams[0]);
        }
      },
      error: err => console.error('[pengaturan] load cameras error:', err),
    });
  }

  get floors(): string[] {
    const set = new Set(this.cameras.map(c => c.floor).filter((f): f is string => !!f));
    return Array.from(set).sort((a, b) => a.localeCompare(b));
  }

  get filteredCameras(): Camera[] {
    return this.cameras.filter(c => {
      const matchFloor  = this.floorFilter === 'semua' || c.floor === this.floorFilter;
      const matchStatus =
        this.statusFilter === 'semua' ||
        (this.statusFilter === 'aktif'    && c.is_active) ||
        (this.statusFilter === 'nonaktif' && !c.is_active);
      return matchFloor && matchStatus;
    });
  }

  onFloorFilterChange(event: Event): void {
    this.floorFilter = (event.target as HTMLSelectElement).value;
  }

  onStatusFilterChange(event: Event): void {
    this.statusFilter = (event.target as HTMLSelectElement).value as 'semua' | 'aktif' | 'nonaktif';
  }

  toggleCamera(cam: Camera, event: Event): void {
    event.stopPropagation();
    this.http.patch<Camera>(`${API}/cameras/${cam.id}/toggle`, {}).subscribe(updated => {
      cam.is_active = updated.is_active;
    });
  }

  deleteCamera(cam: Camera, event: Event): void {
    event.stopPropagation();
    if (!confirm(`Hapus kamera "${cam.name}"?`)) return;
    this.http.delete(`${API}/cameras/${cam.id}`).subscribe(() => {
      this.cameras = this.cameras.filter(c => c.id !== cam.id);
      if (this.selectedCamera?.id === cam.id) {
        this.selectedCamera = null;
        this.snapshotUrl    = '';
        this.savedLines     = [];
        this.pendingLine    = null;
      }
    });
  }

  openAddModal(): void {
    this.editingCamera = null;
    this.formName      = '';
    this.formFloor     = '';
    this.formRtspUrl   = '';
    this.formIsActive  = true;
    this.showModal     = true;
  }

  openEditModal(cam: Camera, event: Event): void {
    event.stopPropagation();
    this.editingCamera = cam;
    this.formName      = cam.name;
    this.formFloor     = cam.floor ?? '';
    this.formRtspUrl   = cam.rtsp_url;
    this.formIsActive  = cam.is_active;
    this.showModal     = true;
  }

  saveCamera(): void {
    const body: Record<string, unknown> = {
      name:     this.formName.trim(),
      floor:    this.formFloor.trim() || null,
      rtsp_url: this.formRtspUrl.trim(),
    };
    if (!this.editingCamera) {
      body['is_active'] = true;
    } else {
      body['is_active'] = this.editingCamera.is_active;
    }
    const req = this.editingCamera
      ? this.http.put<Camera>(`${API}/cameras/${this.editingCamera.id}`, body)
      : this.http.post<Camera>(`${API}/cameras`, body);

    req.subscribe({
      next: () => {
        this.showModal = false;
        this.loadCameras();
      },
      error: (err) => {
        console.error('[pengaturan] saveCamera error:', err);
        alert('Gagal menyimpan kamera. Periksa koneksi ke backend.');
      },
    });
  }

  toggleRtspVisibility(id: number, event: Event): void {
    event.stopPropagation();
    if (this.visibleRtsp.has(id)) {
      this.visibleRtsp.delete(id);
    } else {
      this.visibleRtsp.add(id);
    }
  }

  maskedUrl(url: string): string {
    try {
      const u = new URL(url);
      if (u.password) u.password = '••••••';
      return u.toString();
    } catch {
      return url.replace(/:([^@]+)@/, ':••••••@');
    }
  }

  closeModal(): void { this.showModal = false; }

  // ── Zone config ───────────────────────────────────────────────────────────

  private camId(cam: Camera): string | null {
    return cam.camera_id ?? cam.name ?? null;
  }

  selectCamera(cam: Camera): void {
    const cid = this.camId(cam);
    if (!cid) return;
    this.selectedCamera = cam;
    this.points         = [];
    this.pendingLine    = null;
    this.showZoneForm   = false;
    this.snapshotError  = false;
    this.snapshotUrl    = `${API}/cameras/${cid}/snapshot?t=${Date.now()}`;
    this.loadSavedLines(cid);
  }

  onSelectChange(event: Event): void {
    const id  = parseInt((event.target as HTMLSelectElement).value, 10);
    const cam = this.cameras.find(c => c.id === id);
    if (cam) this.selectCamera(cam);
  }

  onSnapshotLoad(): void {
    this.snapshotError = false;
    const img    = this.imgRef.nativeElement;
    const canvas = this.canvasRef.nativeElement;
    canvas.width  = img.naturalWidth  || img.offsetWidth;
    canvas.height = img.naturalHeight || img.offsetHeight;
    this.drawCanvas();
  }

  onSnapshotError(): void {
    this.snapshotError = true;
  }

  loadSavedLines(cameraId: string): void {
    this.http.get<CrossingLine[]>(`${API}/cameras/${cameraId}/lines`).subscribe({
      next: lines => {
        this.savedLines = lines;
        this.drawCanvas();
      },
    });
  }

  refreshSnapshot(): void {
    if (!this.selectedCamera) return;
    const cid = this.camId(this.selectedCamera);
    if (!cid) return;
    this.snapshotError = false;
    this.snapshotUrl   = `${API}/cameras/${cid}/snapshot?t=${Date.now()}`;
  }

  onCanvasClick(event: MouseEvent): void {
    const canvas = this.canvasRef.nativeElement;
    const img    = this.imgRef.nativeElement;
    if (!img.naturalWidth) return;

    const rect  = canvas.getBoundingClientRect();
    const scaleX = img.naturalWidth  / rect.width;
    const scaleY = img.naturalHeight / rect.height;
    const x = Math.round((event.clientX - rect.left) * scaleX);
    const y = Math.round((event.clientY - rect.top)  * scaleY);

    this.points.push({ x, y });

    if (this.points.length >= 2) {
      this.pendingLine = { p1: this.points[0], p2: this.points[1], inSign: 1 };
      this.points = [];
    }

    this.drawCanvas();
  }

  drawCanvas(): void {
    const canvas = this.canvasRef?.nativeElement;
    if (!canvas) return;
    const img = this.imgRef?.nativeElement;
    if (!img?.naturalWidth) return;

    const ctx = canvas.getContext('2d')!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const lw       = Math.max(4, img.naturalWidth / 300);
    const dotR     = lw * 2.5;
    const fontSize = Math.max(18, img.naturalWidth / 40);

    // Saved lines with IN/OUT arrows
    this.savedLines.forEach((line, i) => {
      const color = LINE_COLORS[i % LINE_COLORS.length];
      ctx.strokeStyle = color;
      ctx.lineWidth   = lw;
      ctx.shadowColor = color;
      ctx.shadowBlur  = 6;
      ctx.beginPath();
      ctx.moveTo(line.p1_x, line.p1_y);
      ctx.lineTo(line.p2_x, line.p2_y);
      ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.fillStyle  = color;
      [{ x: line.p1_x, y: line.p1_y }, { x: line.p2_x, y: line.p2_y }].forEach(p => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, dotR, 0, Math.PI * 2);
        ctx.fill();
      });
      this._drawDirectionArrow(
        ctx, line.p1_x, line.p1_y, line.p2_x, line.p2_y,
        line.in_sign, color, lw, fontSize,
      );
    });

    // First click point (awaiting second)
    if (this.points[0]) {
      ctx.fillStyle = 'rgba(255,255,255,0.9)';
      ctx.beginPath();
      ctx.arc(this.points[0].x, this.points[0].y, dotR, 0, Math.PI * 2);
      ctx.fill();
    }

    // Pending line (dashed white) with IN/OUT arrows
    if (this.pendingLine) {
      const { p1, p2, inSign } = this.pendingLine;
      const dash = Math.round(img.naturalWidth / 60);
      ctx.strokeStyle = 'rgba(255,255,255,0.9)';
      ctx.lineWidth   = lw;
      ctx.setLineDash([dash, Math.round(dash * 0.6)]);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(255,255,255,0.9)';
      [p1, p2].forEach(p => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, dotR, 0, Math.PI * 2);
        ctx.fill();
      });
      this._drawDirectionArrow(
        ctx, p1.x, p1.y, p2.x, p2.y,
        inSign, 'rgba(255,255,255,0.9)', lw, fontSize,
      );
    }
  }

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

    // Perpendicular unit vector pointing to the IN side.
    // _cross_side gradient w.r.t (px,py) = (-dy, dx) → that's the "positive" side.
    const perpX = (-dy / len) * inSign;
    const perpY = ( dx / len) * inSign;

    const mx = (x1 + x2) / 2;
    const my = (y1 + y2) / 2;
    const arrowLen  = Math.max(40, len * 0.25);
    const headLen   = arrowLen * 0.35;
    const ax = mx + perpX * arrowLen;
    const ay = my + perpY * arrowLen;
    const angle = Math.atan2(perpY, perpX);

    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle   = color;
    ctx.lineWidth   = lw * 0.8;
    ctx.shadowBlur  = 0;

    // Shaft
    ctx.beginPath();
    ctx.moveTo(mx, my);
    ctx.lineTo(ax, ay);
    ctx.stroke();

    // Arrowhead
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(ax - headLen * Math.cos(angle - Math.PI / 6), ay - headLen * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(ax - headLen * Math.cos(angle + Math.PI / 6), ay - headLen * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fill();

    // "IN" label
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

  flipInSign(): void {
    if (!this.pendingLine) return;
    this.pendingLine = { ...this.pendingLine, inSign: -this.pendingLine.inSign };
    this.drawCanvas();
  }

  addZone(): void {
    if (!this.pendingLine) return;
    this.showZoneForm = true;
  }

  clearZone(): void {
    this.points      = [];
    this.pendingLine = null;
    this.showZoneForm = false;
    this.roomName    = '';
    this.drawCanvas();
  }

  saveZone(): void {
    const cam = this.selectedCamera;
    const cid = cam ? this.camId(cam) : null;
    if (!cam || !cid || !this.pendingLine) return;

    const newLine = {
      p1_x: this.pendingLine.p1.x, p1_y: this.pendingLine.p1.y,
      p2_x: this.pendingLine.p2.x, p2_y: this.pendingLine.p2.y,
      in_sign: this.pendingLine.inSign,
    };
    const allLines = [
      ...this.savedLines.map(l => ({
        p1_x: l.p1_x, p1_y: l.p1_y,
        p2_x: l.p2_x, p2_y: l.p2_y,
        in_sign: l.in_sign,
      })),
      newLine,
    ];

    this.http.post(`${API}/cameras/${cid}/lines`, allLines).subscribe(() => {
      if (this.roomName.trim()) {
        this.http.post(`${API}/cameras/${cid}/zone`, {
          room_name: this.roomName.trim(),
          floor:     cam.floor ?? null,
        }).subscribe();
      }
      this.pendingLine  = null;
      this.showZoneForm = false;
      this.roomName     = '';
      this.loadSavedLines(cid);
    });
  }

  deleteAllLines(): void {
    const cam = this.selectedCamera;
    const cid = cam ? this.camId(cam) : null;
    if (!cam || !cid) return;
    this.http.post(`${API}/cameras/${cid}/lines`, []).subscribe(() => {
      this.savedLines  = [];
      this.pendingLine = null;
      this.points      = [];
      this.drawCanvas();
    });
  }

  lineColor(i: number): string {
    return LINE_COLORS[i % LINE_COLORS.length];
  }

  formInput(field: 'name' | 'floor' | 'rtspUrl', event: Event): void {
    const val = (event.target as HTMLInputElement).value;
    if (field === 'name')    this.formName    = val;
    if (field === 'floor')   this.formFloor   = val;
    if (field === 'rtspUrl') this.formRtspUrl = val;
  }
}

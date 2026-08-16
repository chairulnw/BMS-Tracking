import { Component, DestroyRef, inject, OnInit } from '@angular/core';
import { NgClass } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { interval } from 'rxjs';
import { startWith, switchMap } from 'rxjs/operators';
import { AuthUrlPipe } from '../../pipes/auth-url.pipe';

const API = 'http://localhost:8002';

interface Stats {
  cameras_total:    number;
  cameras_active:   number;
  cameras_inactive: number;
  events_today:     number;
  detections_today: number;
  persons_today:    number;
  persons_known:    number;
  persons_unknown:  number;
}

interface OccupancyRoom {
  camera_id:         string;
  room_name:         string;
  floor:             string | null;
  current_occupancy: number;
}

interface Person {
  id:                 number;
  name:               string;
  label:              string;
  is_known:           boolean;
  last_camera:        string | null;
  best_thumbnail_url: string | null;
}

interface Camera {
  id:        number;
  camera_id: string | null;
  name:      string;
}

interface CameraEvent {
  id:           number;
  camera_id:    string;
  camera_name:  string | null;
  event_type:   string;
  category:     string;
  description:  string | null;
  snapshot_url: string | null;
  timestamp:    string;
}

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [NgClass, AuthUrlPipe],
  templateUrl: './overview.html',
  styleUrl: './overview.css',
})
export class Dashboard implements OnInit {
  private http       = inject(HttpClient);
  private destroyRef = inject(DestroyRef);

  stats: Stats = {
    cameras_total: 0, cameras_active: 0, cameras_inactive: 0,
    events_today: 0, detections_today: 0, persons_today: 0,
    persons_known: 0, persons_unknown: 0,
  };

  occupancy: OccupancyRoom[] = [];
  persons:   Person[]        = [];
  cameras:   Camera[]        = [];

  events:      CameraEvent[] = [];
  eventsTotal  = 0;
  eventsPages  = 1;
  eventsPage   = 1;
  eventsLimit  = 4;

  filterDate       = this._todayISO();
  filterCamera     = '';
  filterEventType  = '';
  filterCategory   = '';

  readonly EVENT_TYPES = [
    'person_detected', 'zone_entry', 'camera_offline', 'camera_online', 'after_hours_activity',
  ];

  ngOnInit(): void {
    this._loadCameras();

    interval(30000).pipe(
      startWith(0),
      switchMap(() => this.http.get<Stats>(`${API}/stats/today`)),
      takeUntilDestroyed(this.destroyRef),
    ).subscribe({ next: s => this.stats = s, error: () => {} });

    interval(30000).pipe(
      startWith(0),
      switchMap(() => this.http.get<OccupancyRoom[]>(`${API}/occupancy?date=${this._todayISO()}`)),
      takeUntilDestroyed(this.destroyRef),
    ).subscribe({ next: d => this.occupancy = d, error: () => {} });

    interval(30000).pipe(
      startWith(0),
      switchMap(() => this.http.get<Person[]>(`${API}/persons?date=${this._todayISO()}`)),
      takeUntilDestroyed(this.destroyRef),
    ).subscribe({ next: d => this.persons = d.slice(0, 5), error: () => {} });

    this._loadEvents();
  }

  private _loadCameras(): void {
    this.http.get<Camera[]>(`${API}/cameras`).subscribe({
      next: d => this.cameras = d,
    });
  }

  applyFilter(): void {
    this.eventsPage = 1;
    this._loadEvents();
  }

  goToPage(p: number): void {
    if (p < 1 || p > this.eventsPages) return;
    this.eventsPage = p;
    this._loadEvents();
  }

  get pageNumbers(): number[] {
    return Array.from({ length: this.eventsPages }, (_, i) => i + 1);
  }

  get topOccupancy(): OccupancyRoom[] {
    return [...this.occupancy]
      .sort((a, b) => b.current_occupancy - a.current_occupancy)
      .slice(0, 6);
  }

  get roomSummary(): { total: number; occupied: number; empty: number } {
    const byRoom = new Map<string, number>();
    for (const r of this.occupancy) {
      byRoom.set(r.room_name, (byRoom.get(r.room_name) ?? 0) + r.current_occupancy);
    }
    const total    = byRoom.size;
    const occupied = [...byRoom.values()].filter(v => v > 0).length;
    return { total, occupied, empty: total - occupied };
  }

  cameraName(camera_id: string | null): string {
    if (!camera_id) return '—';
    return this.cameras.find(c => c.camera_id === camera_id)?.name ?? camera_id;
  }

  private _loadEvents(): void {
    const params = new URLSearchParams({
      date:  this.filterDate,
      page:  String(this.eventsPage),
      limit: String(this.eventsLimit),
    });
    if (this.filterCamera)    params.set('camera_id',  this.filterCamera);
    if (this.filterEventType) params.set('event_type', this.filterEventType);
    if (this.filterCategory)  params.set('category',   this.filterCategory);

    this.http.get<{ events: CameraEvent[]; total: number; pages: number }>(
      `${API}/camera-events?${params}`
    ).subscribe({
      next: res => {
        this.events      = res.events;
        this.eventsTotal = res.total;
        this.eventsPages = res.pages;
      },
      error: () => {},
    });
  }

  private _todayISO(): string {
    const d = new Date();
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
  }

  fmtTime(iso: string): string {
    return new Date(iso).toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }

  fmtDate(iso: string): string {
    return new Date(iso).toLocaleDateString('id-ID', { day: 'numeric', month: 'long', year: 'numeric' });
  }

  categoryClass(cat: string): string {
    if (cat === 'critical') return 'badge--critical';
    if (cat === 'warning')  return 'badge--warning';
    return 'badge--info';
  }
}

import { Routes } from '@angular/router';
import { Dashboard } from './pages/overview/overview';
import { Liveview } from './pages/liveview/liveview';
import { Playback } from './pages/playback/playback';
import { Eventlog } from './pages/eventlog/eventlog';
import { CameraPage } from './pages/camera/camera';
import { ZonePage } from './pages/zone/zone';
import { People } from './pages/people/people';
import { Login } from './pages/login/login';
import { authGuard } from './guards/auth.guard';

export const routes: Routes = [
  { path: 'login', component: Login },
  { path: '', component: Dashboard, canActivate: [authGuard] },
  { path: 'dashboard', component: Dashboard, canActivate: [authGuard] },
  { path: 'overview', component: Dashboard, canActivate: [authGuard] },
  { path: 'liveview', component: Liveview, canActivate: [authGuard] },
  { path: 'playback', component: Playback, canActivate: [authGuard] },
  { path: 'eventlog', component: Eventlog, canActivate: [authGuard] },
  { path: 'camera', component: CameraPage, canActivate: [authGuard] },
  { path: 'zone', component: ZonePage, canActivate: [authGuard] },
  { path: 'people', component: People, canActivate: [authGuard] },
];
